// ignore-tidy-filelength
//! "Late resolution" is the pass that resolves most of names in a crate beside imports and macros.
//! It runs when the crate is fully expanded and its module structure is fully built.
//! So it just walks through the crate and resolves all the expressions, types, etc.
//!
//! If you wonder why there's no `early.rs`, that's because it's split into three files -
//! `build_reduced_graph.rs`, `macros.rs` and `imports.rs`.

use RibKind::*;

use crate::{path_names_to_string, rustdoc, BindingError, Finalize, LexicalScopeBinding};
use crate::{Module, ModuleOrUniformRoot, NameBinding, ParentScope, PathResult};
use crate::{ResolutionError, Resolver, Segment, UseError};

use rustc_ast::ptr::P;
use rustc_ast::visit::{self, AssocCtxt, BoundKind, FnCtxt, FnKind, Visitor};
use rustc_ast::*;
use rustc_data_structures::fx::{FxHashMap, FxHashSet, FxIndexMap};
use rustc_errors::{Applicability, DiagnosticArgValue, DiagnosticId, IntoDiagnosticArg};
use rustc_hir::def::Namespace::{self, *};
use rustc_hir::def::{self, CtorKind, DefKind, LifetimeRes, PartialRes, PerNS};
use rustc_hir::def_id::{DefId, LocalDefId, CRATE_DEF_ID, LOCAL_CRATE};
use rustc_hir::{BindingAnnotation, PrimTy, TraitCandidate};
use rustc_middle::middle::resolve_bound_vars::Set1;
use rustc_middle::{bug, span_bug};
use rustc_session::config::{CrateType, ResolveDocLinks};
use rustc_session::lint;
use rustc_span::source_map::{respan, Spanned};
use rustc_span::symbol::{kw, sym, Ident, Symbol};
use rustc_span::{BytePos, Span, SyntaxContext};
use smallvec::{smallvec, SmallVec};

use std::assert_matches::debug_assert_matches;
use std::borrow::Cow;
use std::collections::{hash_map::Entry, BTreeSet};
use std::mem::{replace, swap, take};

mod diagnostics;

type Res = def::Res<NodeId>;

type IdentMap<T> = FxHashMap<Ident, T>;

/// Map from the name in a pattern to its binding mode.
type BindingMap = IdentMap<BindingInfo>;

use diagnostics::{
    ElisionFnParameter, LifetimeElisionCandidate, MissingLifetime, MissingLifetimeKind,
};

#[derive(Copy, Clone, Debug)]
struct BindingInfo {
    span: Span,
    annotation: BindingAnnotation,
}

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub(crate) enum PatternSource {
    Match,
    Let,
    For,
    FnParam,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum IsRepeatExpr {
    No,
    Yes,
}

impl PatternSource {
    fn descr(self) -> &'static str {
        match self {
            PatternSource::Match => "match binding",
            PatternSource::Let => "let binding",
            PatternSource::For => "for binding",
            PatternSource::FnParam => "function parameter",
        }
    }
}

impl IntoDiagnosticArg for PatternSource {
    fn into_diagnostic_arg(self) -> DiagnosticArgValue<'static> {
        DiagnosticArgValue::Str(Cow::Borrowed(self.descr()))
    }
}

/// Denotes whether the context for the set of already bound bindings is a `Product`
/// or `Or` context. This is used in e.g., `fresh_binding` and `resolve_pattern_inner`.
/// See those functions for more information.
#[derive(PartialEq)]
enum PatBoundCtx {
    /// A product pattern context, e.g., `Variant(a, b)`.
    Product,
    /// An or-pattern context, e.g., `p_0 | ... | p_n`.
    Or,
}

/// Does this the item (from the item rib scope) allow generic parameters?
#[derive(Copy, Clone, Debug)]
pub(crate) enum HasGenericParams {
    Yes(Span),
    No,
}

/// May this constant have generics?
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub(crate) enum ConstantHasGenerics {
    Yes,
    No,
}

impl ConstantHasGenerics {
    fn force_yes_if(self, b: bool) -> Self {
        if b { Self::Yes } else { self }
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub(crate) enum ConstantItemKind {
    Const,
    Static,
}

/// The rib kind restricts certain accesses,
/// e.g. to a `Res::Local` of an outer item.
#[derive(Copy, Clone, Debug)]
pub(crate) enum RibKind<'a> {
    /// No restriction needs to be applied.
    NormalRibKind,

    /// We passed through an impl or trait and are now in one of its
    /// methods or associated types. Allow references to ty params that impl or trait
    /// binds. Disallow any other upvars (including other ty params that are
    /// upvars).
    AssocItemRibKind,

    /// We passed through a closure. Disallow labels.
    ClosureOrAsyncRibKind,

    /// We passed through an item scope. Disallow upvars.
    ItemRibKind(HasGenericParams),

    /// We're in a constant item. Can't refer to dynamic stuff.
    ///
    /// The item may reference generic parameters in trivial constant expressions.
    /// All other constants aren't allowed to use generic params at all.
    ConstantItemRibKind(ConstantHasGenerics, Option<(Ident, ConstantItemKind)>),

    /// We passed through a module.
    ModuleRibKind(Module<'a>),

    /// We passed through a `macro_rules!` statement
    MacroDefinition(DefId),

    /// All bindings in this rib are generic parameters that can't be used
    /// from the default of a generic parameter because they're not declared
    /// before said generic parameter. Also see the `visit_generics` override.
    ForwardGenericParamBanRibKind,

    /// We are inside of the type of a const parameter. Can't refer to any
    /// parameters.
    ConstParamTyRibKind,

    /// We are inside a `sym` inline assembly operand. Can only refer to
    /// globals.
    InlineAsmSymRibKind,
}

impl RibKind<'_> {
    /// Whether this rib kind contains generic parameters, as opposed to local
    /// variables.
    pub(crate) fn contains_params(&self) -> bool {
        match self {
            NormalRibKind
            | ClosureOrAsyncRibKind
            | ConstantItemRibKind(..)
            | ModuleRibKind(_)
            | MacroDefinition(_)
            | ConstParamTyRibKind
            | InlineAsmSymRibKind => false,
            AssocItemRibKind | ItemRibKind(_) | ForwardGenericParamBanRibKind => true,
        }
    }

    /// This rib forbids referring to labels defined in upwards ribs.
    fn is_label_barrier(self) -> bool {
        match self {
            NormalRibKind | MacroDefinition(..) => false,

            AssocItemRibKind
            | ClosureOrAsyncRibKind
            | ItemRibKind(..)
            | ConstantItemRibKind(..)
            | ModuleRibKind(..)
            | ForwardGenericParamBanRibKind
            | ConstParamTyRibKind
            | InlineAsmSymRibKind => true,
        }
    }
}

/// A single local scope.
///
/// A rib represents a scope names can live in. Note that these appear in many places, not just
/// around braces. At any place where the list of accessible names (of the given namespace)
/// changes or a new restrictions on the name accessibility are introduced, a new rib is put onto a
/// stack. This may be, for example, a `let` statement (because it introduces variables), a macro,
/// etc.
///
/// Different [rib kinds](enum@RibKind) are transparent for different names.
///
/// The resolution keeps a separate stack of ribs as it traverses the AST for each namespace. When
/// resolving, the name is looked up from inside out.
#[derive(Debug)]
pub(crate) struct Rib<'a, R = Res> {
    pub bindings: IdentMap<R>,
    pub kind: RibKind<'a>,
}

impl<'a, R> Rib<'a, R> {
    fn new(kind: RibKind<'a>) -> Rib<'a, R> {
        Rib { bindings: Default::default(), kind }
    }
}

#[derive(Clone, Copy, Debug)]
enum LifetimeUseSet {
    One { use_span: Span, use_ctxt: visit::LifetimeCtxt },
    Many,
}

#[derive(Copy, Clone, Debug)]
enum LifetimeRibKind {
    // -- Ribs introducing named lifetimes
    //
    /// This rib declares generic parameters.
    /// Only for this kind the `LifetimeRib::bindings` field can be non-empty.
    Generics { binder: NodeId, span: Span, kind: LifetimeBinderKind },

    // -- Ribs introducing unnamed lifetimes
    //
    /// Create a new anonymous lifetime parameter and reference it.
    ///
    /// If `report_in_path`, report an error when encountering lifetime elision in a path:
    /// ```compile_fail
    /// struct Foo<'a> { x: &'a () }
    /// async fn foo(x: Foo) {}
    /// ```
    ///
    /// Note: the error should not trigger when the elided lifetime is in a pattern or
    /// expression-position path:
    /// ```
    /// struct Foo<'a> { x: &'a () }
    /// async fn foo(Foo { x: _ }: Foo<'_>) {}
    /// ```
    AnonymousCreateParameter { binder: NodeId, report_in_path: bool },

    /// Replace all anonymous lifetimes by provided lifetime.
    Elided(LifetimeRes),

    // -- Barrier ribs that stop lifetime lookup, or continue it but produce an error later.
    //
    /// Give a hard error when either `&` or `'_` is written. Used to
    /// rule out things like `where T: Foo<'_>`. Does not imply an
    /// error on default object bounds (e.g., `Box<dyn Foo>`).
    AnonymousReportError,

    /// Signal we cannot find which should be the anonymous lifetime.
    ElisionFailure,

    /// FIXME(const_generics): This patches over an ICE caused by non-'static lifetimes in const
    /// generics. We are disallowing this until we can decide on how we want to handle non-'static
    /// lifetimes in const generics. See issue #74052 for discussion.
    ConstGeneric,

    /// Non-static lifetimes are prohibited in anonymous constants under `min_const_generics`.
    /// This function will emit an error if `generic_const_exprs` is not enabled, the body
    /// identified by `body_id` is an anonymous constant and `lifetime_ref` is non-static.
    AnonConst,

    /// This rib acts as a barrier to forbid reference to lifetimes of a parent item.
    Item,
}

#[derive(Copy, Clone, Debug)]
enum LifetimeBinderKind {
    BareFnType,
    PolyTrait,
    WhereBound,
    Item,
    Function,
    Closure,
    ImplBlock,
}

impl LifetimeBinderKind {
    fn descr(self) -> &'static str {
        use LifetimeBinderKind::*;
        match self {
            BareFnType => "type",
            PolyTrait => "bound",
            WhereBound => "bound",
            Item => "item",
            ImplBlock => "impl block",
            Function => "function",
            Closure => "closure",
        }
    }
}

#[derive(Debug)]
struct LifetimeRib {
    kind: LifetimeRibKind,
    // We need to preserve insertion order for async fns.
    bindings: FxIndexMap<Ident, (NodeId, LifetimeRes)>,
}

impl LifetimeRib {
    fn new(kind: LifetimeRibKind) -> LifetimeRib {
        LifetimeRib { bindings: Default::default(), kind }
    }
}

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub(crate) enum AliasPossibility {
    No,
    Maybe,
}

#[derive(Copy, Clone, Debug)]
pub(crate) enum PathSource<'a> {
    // Type paths `Path`.
    Type,
    // Trait paths in bounds or impls.
    Trait(AliasPossibility),
    // Expression paths `path`, with optional parent context.
    Expr(Option<&'a Expr>),
    // Paths in path patterns `Path`.
    Pat,
    // Paths in struct expressions and patterns `Path { .. }`.
    Struct,
    // Paths in tuple struct patterns `Path(..)`.
    TupleStruct(Span, &'a [Span]),
    // `m::A::B` in `<T as m::A>::B::C`.
    TraitItem(Namespace),
}

impl<'a> PathSource<'a> {
    fn namespace(self) -> Namespace {
        match self {
            PathSource::Type | PathSource::Trait(_) | PathSource::Struct => TypeNS,
            PathSource::Expr(..) | PathSource::Pat | PathSource::TupleStruct(..) => ValueNS,
            PathSource::TraitItem(ns) => ns,
        }
    }

    fn defer_to_typeck(self) -> bool {
        match self {
            PathSource::Type
            | PathSource::Expr(..)
            | PathSource::Pat
            | PathSource::Struct
            | PathSource::TupleStruct(..) => true,
            PathSource::Trait(_) | PathSource::TraitItem(..) => false,
        }
    }

    fn descr_expected(self) -> &'static str {
        match &self {
            PathSource::Type => "type",
            PathSource::Trait(_) => "trait",
            PathSource::Pat => "unit struct, unit variant or constant",
            PathSource::Struct => "struct, variant or union type",
            PathSource::TupleStruct(..) => "tuple struct or tuple variant",
            PathSource::TraitItem(ns) => match ns {
                TypeNS => "associated type",
                ValueNS => "method or associated constant",
                MacroNS => bug!("associated macro"),
            },
            PathSource::Expr(parent) => match parent.as_ref().map(|p| &p.kind) {
                // "function" here means "anything callable" rather than `DefKind::Fn`,
                // this is not precise but usually more helpful than just "value".
                Some(ExprKind::Call(call_expr, _)) => match &call_expr.kind {
                    // the case of `::some_crate()`
                    ExprKind::Path(_, path)
                        if path.segments.len() == 2
                            && path.segments[0].ident.name == kw::PathRoot =>
                    {
                        "external crate"
                    }
                    ExprKind::Path(_, path) => {
                        let mut msg = "function";
                        if let Some(segment) = path.segments.iter().last() {
                            if let Some(c) = segment.ident.to_string().chars().next() {
                                if c.is_uppercase() {
                                    msg = "function, tuple struct or tuple variant";
                                }
                            }
                        }
                        msg
                    }
                    _ => "function",
                },
                _ => "value",
            },
        }
    }

    fn is_call(self) -> bool {
        matches!(self, PathSource::Expr(Some(&Expr { kind: ExprKind::Call(..), .. })))
    }

    pub(crate) fn is_expected(self, res: Res) -> bool {
        match self {
            PathSource::Type => matches!(
                res,
                Res::Def(
                    DefKind::Struct
                        | DefKind::Union
                        | DefKind::Enum
                        | DefKind::Trait
                        | DefKind::TraitAlias
                        | DefKind::TyAlias
                        | DefKind::AssocTy
                        | DefKind::TyParam
                        | DefKind::OpaqueTy
                        | DefKind::ForeignTy,
                    _,
                ) | Res::PrimTy(..)
                    | Res::SelfTyParam { .. }
                    | Res::SelfTyAlias { .. }
            ),
            PathSource::Trait(AliasPossibility::No) => matches!(res, Res::Def(DefKind::Trait, _)),
            PathSource::Trait(AliasPossibility::Maybe) => {
                matches!(res, Res::Def(DefKind::Trait | DefKind::TraitAlias, _))
            }
            PathSource::Expr(..) => matches!(
                res,
                Res::Def(
                    DefKind::Ctor(_, CtorKind::Const | CtorKind::Fn)
                        | DefKind::Const
                        | DefKind::Static(_)
                        | DefKind::Fn
                        | DefKind::AssocFn
                        | DefKind::AssocConst
                        | DefKind::ConstParam,
                    _,
                ) | Res::Local(..)
                    | Res::SelfCtor(..)
            ),
            PathSource::Pat => {
                res.expected_in_unit_struct_pat()
                    || matches!(res, Res::Def(DefKind::Const | DefKind::AssocConst, _))
            }
            PathSource::TupleStruct(..) => res.expected_in_tuple_struct_pat(),
            PathSource::Struct => matches!(
                res,
                Res::Def(
                    DefKind::Struct
                        | DefKind::Union
                        | DefKind::Variant
                        | DefKind::TyAlias
                        | DefKind::AssocTy,
                    _,
                ) | Res::SelfTyParam { .. }
                    | Res::SelfTyAlias { .. }
            ),
            PathSource::TraitItem(ns) => match res {
                Res::Def(DefKind::AssocConst | DefKind::AssocFn, _) if ns == ValueNS => true,
                Res::Def(DefKind::AssocTy, _) if ns == TypeNS => true,
                _ => false,
            },
        }
    }

    fn error_code(self, has_unexpected_resolution: bool) -> DiagnosticId {
        use rustc_errors::error_code;
        match (self, has_unexpected_resolution) {
            (PathSource::Trait(_), true) => error_code!(E0404),
            (PathSource::Trait(_), false) => error_code!(E0405),
            (PathSource::Type, true) => error_code!(E0573),
            (PathSource::Type, false) => error_code!(E0412),
            (PathSource::Struct, true) => error_code!(E0574),
            (PathSource::Struct, false) => error_code!(E0422),
            (PathSource::Expr(..), true) => error_code!(E0423),
            (PathSource::Expr(..), false) => error_code!(E0425),
            (PathSource::Pat | PathSource::TupleStruct(..), true) => error_code!(E0532),
            (PathSource::Pat | PathSource::TupleStruct(..), false) => error_code!(E0531),
            (PathSource::TraitItem(..), true) => error_code!(E0575),
            (PathSource::TraitItem(..), false) => error_code!(E0576),
        }
    }
}

/// At this point for most items we can answer whether that item is exported or not,
/// but some items like impls require type information to determine exported-ness, so we make a
/// conservative estimate for them (e.g. based on nominal visibility).
#[derive(Clone, Copy)]
enum MaybeExported<'a> {
    Ok(NodeId),
    Impl(Option<DefId>),
    ImplItem(Result<DefId, &'a Visibility>),
}

impl MaybeExported<'_> {
    fn eval(self, r: &Resolver<'_, '_>) -> bool {
        let def_id = match self {
            MaybeExported::Ok(node_id) => Some(r.local_def_id(node_id)),
            MaybeExported::Impl(Some(trait_def_id)) | MaybeExported::ImplItem(Ok(trait_def_id)) => {
                trait_def_id.as_local()
            }
            MaybeExported::Impl(None) => return true,
            MaybeExported::ImplItem(Err(vis)) => return vis.kind.is_pub(),
        };
        def_id.map_or(true, |def_id| r.effective_visibilities.is_exported(def_id))
    }
}

#[derive(Default)]
struct DiagnosticMetadata<'ast> {
    /// The current trait's associated items' ident, used for diagnostic suggestions.
    current_trait_assoc_items: Option<&'ast [P<AssocItem>]>,

    /// The current self type if inside an impl (used for better errors).
    current_self_type: Option<Ty>,

    /// The current self item if inside an ADT (used for better errors).
    current_self_item: Option<NodeId>,

    /// The current trait (used to suggest).
    current_item: Option<&'ast Item>,

    /// When processing generics and encountering a type not found, suggest introducing a type
    /// param.
    currently_processing_generics: bool,

    /// The current enclosing (non-closure) function (used for better errors).
    current_function: Option<(FnKind<'ast>, Span)>,

    /// A list of labels as of yet unused. Labels will be removed from this map when
    /// they are used (in a `break` or `continue` statement)
    unused_labels: FxHashMap<NodeId, Span>,

    /// Only used for better errors on `fn(): fn()`.
    current_type_ascription: Vec<Span>,

    /// Only used for better errors on `let x = { foo: bar };`.
    /// In the case of a parse error with `let x = { foo: bar, };`, this isn't needed, it's only
    /// needed for cases where this parses as a correct type ascription.
    current_block_could_be_bare_struct_literal: Option<Span>,

    /// Only used for better errors on `let <pat>: <expr, not type>;`.
    current_let_binding: Option<(Span, Option<Span>, Option<Span>)>,

    /// Used to detect possible `if let` written without `let` and to provide structured suggestion.
    in_if_condition: Option<&'ast Expr>,

    /// Used to detect possible new binding written without `let` and to provide structured suggestion.
    in_assignment: Option<&'ast Expr>,
    is_assign_rhs: bool,

    /// Used to detect possible `.` -> `..` typo when calling methods.
    in_range: Option<(&'ast Expr, &'ast Expr)>,

    /// If we are currently in a trait object definition. Used to point at the bounds when
    /// encountering a struct or enum.
    current_trait_object: Option<&'ast [ast::GenericBound]>,

    /// Given `where <T as Bar>::Baz: String`, suggest `where T: Bar<Baz = String>`.
    current_where_predicate: Option<&'ast WherePredicate>,

    current_type_path: Option<&'ast Ty>,

    /// The current impl items (used to suggest).
    current_impl_items: Option<&'ast [P<AssocItem>]>,

    /// When processing impl trait
    currently_processing_impl_trait: Option<(TraitRef, Ty)>,

    /// Accumulate the errors due to missed lifetime elision,
    /// and report them all at once for each function.
    current_elision_failures: Vec<MissingLifetime>,
}

struct LateResolutionVisitor<'a, 'b, 'ast, 'tcx> {
    r: &'b mut Resolver<'a, 'tcx>,

    /// The module that represents the current item scope.
    parent_scope: ParentScope<'a>,

    /// The current set of local scopes for types and values.
    /// FIXME #4948: Reuse ribs to avoid allocation.
    ribs: PerNS<Vec<Rib<'a>>>,

    /// Previous poped `rib`, only used for diagnostic.
    last_block_rib: Option<Rib<'a>>,

    /// The current set of local scopes, for labels.
    label_ribs: Vec<Rib<'a, NodeId>>,

    /// The current set of local scopes for lifetimes.
    lifetime_ribs: Vec<LifetimeRib>,

    /// We are looking for lifetimes in an elision context.
    /// The set contains all the resolutions that we encountered so far.
    /// They will be used to determine the correct lifetime for the fn return type.
    /// The `LifetimeElisionCandidate` is used for diagnostics, to suggest introducing named
    /// lifetimes.
    lifetime_elision_candidates: Option<Vec<(LifetimeRes, LifetimeElisionCandidate)>>,

    /// The trait that the current context can refer to.
    current_trait_ref: Option<(Module<'a>, TraitRef)>,

    /// Fields used to add information to diagnostic errors.
    diagnostic_metadata: Box<DiagnosticMetadata<'ast>>,

    /// State used to know whether to ignore resolution errors for function bodies.
    ///
    /// In particular, rustdoc uses this to avoid giving errors for `cfg()` items.
    /// In most cases this will be `None`, in which case errors will always be reported.
    /// If it is `true`, then it will be updated when entering a nested function or trait body.
    in_func_body: bool,

    /// Count the number of places a lifetime is used.
    lifetime_uses: FxHashMap<LocalDefId, LifetimeUseSet>,
}

/// Walks the whole crate in DFS order, visiting each item, resolving names as it goes.
impl<'a: 'ast, 'ast, 'tcx> Visitor<'ast> for LateResolutionVisitor<'a, '_, 'ast, 'tcx> {
    fn visit_attribute(&mut self, _: &'ast Attribute) {
        // We do not want to resolve expressions that appear in attributes,
        // as they do not correspond to actual code.
    }
    fn visit_item(&mut self, item: &'ast Item) {
        let prev = replace(&mut self.diagnostic_metadata.current_item, Some(item));
        // Always report errors in items we just entered.
        let old_ignore = replace(&mut self.in_func_body, false);
        self.with_lifetime_rib(LifetimeRibKind::Item, |this| this.resolve_item(item));
        self.in_func_body = old_ignore;
        self.diagnostic_metadata.current_item = prev;
    }
    fn visit_arm(&mut self, arm: &'ast Arm) {
        self.resolve_arm(arm);
    }
    fn visit_block(&mut self, block: &'ast Block) {
        let old_macro_rules = self.parent_scope.macro_rules;
        self.resolve_block(block);
        self.parent_scope.macro_rules = old_macro_rules;
    }
    fn visit_anon_const(&mut self, constant: &'ast AnonConst) {
        // We deal with repeat expressions explicitly in `resolve_expr`.
        self.with_lifetime_rib(LifetimeRibKind::AnonConst, |this| {
            this.with_lifetime_rib(LifetimeRibKind::Elided(LifetimeRes::Static), |this| {
                this.resolve_anon_const(constant, IsRepeatExpr::No);
            })
        })
    }
    fn visit_expr(&mut self, expr: &'ast Expr) {
        self.resolve_expr(expr, None);
    }
    fn visit_local(&mut self, local: &'ast Local) {
        let local_spans = match local.pat.kind {
            // We check for this to avoid tuple struct fields.
            PatKind::Wild => None,
            _ => Some((
                local.pat.span,
                local.ty.as_ref().map(|ty| ty.span),
                local.kind.init().map(|init| init.span),
            )),
        };
        let original = replace(&mut self.diagnostic_metadata.current_let_binding, local_spans);
        self.resolve_local(local);
        self.diagnostic_metadata.current_let_binding = original;
    }
    fn visit_ty(&mut self, ty: &'ast Ty) {
        let prev = self.diagnostic_metadata.current_trait_object;
        let prev_ty = self.diagnostic_metadata.current_type_path;
        match ty.kind {
            TyKind::Ref(None, _) => {
                // Elided lifetime in reference: we resolve as if there was some lifetime `'_` with
                // NodeId `ty.id`.
                // This span will be used in case of elision failure.
                let span = self.r.tcx.sess.source_map().start_point(ty.span);
                self.resolve_elided_lifetime(ty.id, span);
                visit::walk_ty(self, ty);
            }
            TyKind::Path(ref qself, ref path) => {
                self.diagnostic_metadata.current_type_path = Some(ty);
                self.smart_resolve_path(ty.id, &qself, path, PathSource::Type);

                // Check whether we should interpret this as a bare trait object.
                if qself.is_none()
                    && let Some(partial_res) = self.r.partial_res_map.get(&ty.id)
                    && let Some(Res::Def(DefKind::Trait | DefKind::TraitAlias, _)) = partial_res.full_res()
                {
                    // This path is actually a bare trait object. In case of a bare `Fn`-trait
                    // object with anonymous lifetimes, we need this rib to correctly place the
                    // synthetic lifetimes.
                    let span = ty.span.shrink_to_lo().to(path.span.shrink_to_lo());
                    self.with_generic_param_rib(
                        &[],
                        NormalRibKind,
                        LifetimeRibKind::Generics {
                            binder: ty.id,
                            kind: LifetimeBinderKind::PolyTrait,
                            span,
                        },
                        |this| this.visit_path(&path, ty.id),
                    );
                } else {
                    visit::walk_ty(self, ty)
                }
            }
            TyKind::ImplicitSelf => {
                let self_ty = Ident::with_dummy_span(kw::SelfUpper);
                let res = self
                    .resolve_ident_in_lexical_scope(
                        self_ty,
                        TypeNS,
                        Some(Finalize::new(ty.id, ty.span)),
                        None,
                    )
                    .map_or(Res::Err, |d| d.res());
                self.r.record_partial_res(ty.id, PartialRes::new(res));
                visit::walk_ty(self, ty)
            }
            TyKind::ImplTrait(..) => {
                let candidates = self.lifetime_elision_candidates.take();
                visit::walk_ty(self, ty);
                self.lifetime_elision_candidates = candidates;
            }
            TyKind::TraitObject(ref bounds, ..) => {
                self.diagnostic_metadata.current_trait_object = Some(&bounds[..]);
                visit::walk_ty(self, ty)
            }
            TyKind::BareFn(ref bare_fn) => {
                let span = ty.span.shrink_to_lo().to(bare_fn.decl_span.shrink_to_lo());
                self.with_generic_param_rib(
                    &bare_fn.generic_params,
                    NormalRibKind,
                    LifetimeRibKind::Generics {
                        binder: ty.id,
                        kind: LifetimeBinderKind::BareFnType,
                        span,
                    },
                    |this| {
                        this.visit_generic_params(&bare_fn.generic_params, false);
                        this.with_lifetime_rib(
                            LifetimeRibKind::AnonymousCreateParameter {
                                binder: ty.id,
                                report_in_path: false,
                            },
                            |this| {
                                this.resolve_fn_signature(
                                    ty.id,
                                    false,
                                    // We don't need to deal with patterns in parameters, because
                                    // they are not possible for foreign or bodiless functions.
                                    bare_fn
                                        .decl
                                        .inputs
                                        .iter()
                                        .map(|Param { ty, .. }| (None, &**ty)),
                                    &bare_fn.decl.output,
                                )
                            },
                        );
                    },
                )
            }
            _ => visit::walk_ty(self, ty),
        }
        self.diagnostic_metadata.current_trait_object = prev;
        self.diagnostic_metadata.current_type_path = prev_ty;
    }
    fn visit_poly_trait_ref(&mut self, tref: &'ast PolyTraitRef) {
        let span = tref.span.shrink_to_lo().to(tref.trait_ref.path.span.shrink_to_lo());
        self.with_generic_param_rib(
            &tref.bound_generic_params,
            NormalRibKind,
            LifetimeRibKind::Generics {
                binder: tref.trait_ref.ref_id,
                kind: LifetimeBinderKind::PolyTrait,
                span,
            },
            |this| {
                this.visit_generic_params(&tref.bound_generic_params, false);
                this.smart_resolve_path(
                    tref.trait_ref.ref_id,
                    &None,
                    &tref.trait_ref.path,
                    PathSource::Trait(AliasPossibility::Maybe),
                );
                this.visit_trait_ref(&tref.trait_ref);
            },
        );
    }
    fn visit_foreign_item(&mut self, foreign_item: &'ast ForeignItem) {
        self.resolve_doc_links(&foreign_item.attrs, MaybeExported::Ok(foreign_item.id));
        match foreign_item.kind {
            ForeignItemKind::TyAlias(box TyAlias { ref generics, .. }) => {
                self.with_generic_param_rib(
                    &generics.params,
                    ItemRibKind(HasGenericParams::Yes(generics.span)),
                    LifetimeRibKind::Generics {
                        binder: foreign_item.id,
                        kind: LifetimeBinderKind::Item,
                        span: generics.span,
                    },
                    |this| visit::walk_foreign_item(this, foreign_item),
                );
            }
            ForeignItemKind::Fn(box Fn { ref generics, .. }) => {
                self.with_generic_param_rib(
                    &generics.params,
                    ItemRibKind(HasGenericParams::Yes(generics.span)),
                    LifetimeRibKind::Generics {
                        binder: foreign_item.id,
                        kind: LifetimeBinderKind::Function,
                        span: generics.span,
                    },
                    |this| visit::walk_foreign_item(this, foreign_item),
                );
            }
            ForeignItemKind::Static(..) => {
                self.with_static_rib(|this| {
                    visit::walk_foreign_item(this, foreign_item);
                });
            }
            ForeignItemKind::MacCall(..) => {
                panic!("unexpanded macro in resolve!")
            }
        }
    }
    fn visit_fn(&mut self, fn_kind: FnKind<'ast>, sp: Span, fn_id: NodeId) {
        let previous_value = self.diagnostic_metadata.current_function;
        match fn_kind {
            // Bail if the function is foreign, and thus cannot validly have
            // a body, or if there's no body for some other reason.
            FnKind::Fn(FnCtxt::Foreign, _, sig, _, generics, _)
            | FnKind::Fn(_, _, sig, _, generics, None) => {
                self.visit_fn_header(&sig.header);
                self.visit_generics(generics);
                self.with_lifetime_rib(
                    LifetimeRibKind::AnonymousCreateParameter {
                        binder: fn_id,
                        report_in_path: false,
                    },
                    |this| {
                        this.resolve_fn_signature(
                            fn_id,
                            sig.decl.has_self(),
                            sig.decl.inputs.iter().map(|Param { ty, .. }| (None, &**ty)),
                            &sig.decl.output,
                        );

                        this.record_lifetime_params_for_async(
                            fn_id,
                            sig.header.asyncness.opt_return_id(),
                        );
                    },
                );
                return;
            }
            FnKind::Fn(..) => {
                self.diagnostic_metadata.current_function = Some((fn_kind, sp));
            }
            // Do not update `current_function` for closures: it suggests `self` parameters.
            FnKind::Closure(..) => {}
        };
        debug!("(resolving function) entering function");

        // Create a value rib for the function.
        self.with_rib(ValueNS, ClosureOrAsyncRibKind, |this| {
            // Create a label rib for the function.
            this.with_label_rib(ClosureOrAsyncRibKind, |this| {
                match fn_kind {
                    FnKind::Fn(_, _, sig, _, generics, body) => {
                        this.visit_generics(generics);

                        let declaration = &sig.decl;
                        let async_node_id = sig.header.asyncness.opt_return_id();

                        this.with_lifetime_rib(
                            LifetimeRibKind::AnonymousCreateParameter {
                                binder: fn_id,
                                report_in_path: async_node_id.is_some(),
                            },
                            |this| {
                                this.resolve_fn_signature(
                                    fn_id,
                                    declaration.has_self(),
                                    declaration
                                        .inputs
                                        .iter()
                                        .map(|Param { pat, ty, .. }| (Some(&**pat), &**ty)),
                                    &declaration.output,
                                )
                            },
                        );

                        this.record_lifetime_params_for_async(fn_id, async_node_id);

                        if let Some(body) = body {
                            // Ignore errors in function bodies if this is rustdoc
                            // Be sure not to set this until the function signature has been resolved.
                            let previous_state = replace(&mut this.in_func_body, true);
                            // We only care block in the same function
                            this.last_block_rib = None;
                            // Resolve the function body, potentially inside the body of an async closure
                            this.with_lifetime_rib(
                                LifetimeRibKind::Elided(LifetimeRes::Infer),
                                |this| this.visit_block(body),
                            );

                            debug!("(resolving function) leaving function");
                            this.in_func_body = previous_state;
                        }
                    }
                    FnKind::Closure(binder, declaration, body) => {
                        this.visit_closure_binder(binder);

                        this.with_lifetime_rib(
                            match binder {
                                // We do not have any explicit generic lifetime parameter.
                                ClosureBinder::NotPresent => {
                                    LifetimeRibKind::AnonymousCreateParameter {
                                        binder: fn_id,
                                        report_in_path: false,
                                    }
                                }
                                ClosureBinder::For { .. } => LifetimeRibKind::AnonymousReportError,
                            },
                            // Add each argument to the rib.
                            |this| this.resolve_params(&declaration.inputs),
                        );
                        this.with_lifetime_rib(
                            match binder {
                                ClosureBinder::NotPresent => {
                                    LifetimeRibKind::Elided(LifetimeRes::Infer)
                                }
                                ClosureBinder::For { .. } => LifetimeRibKind::AnonymousReportError,
                            },
                            |this| visit::walk_fn_ret_ty(this, &declaration.output),
                        );

                        // Ignore errors in function bodies if this is rustdoc
                        // Be sure not to set this until the function signature has been resolved.
                        let previous_state = replace(&mut this.in_func_body, true);
                        // Resolve the function body, potentially inside the body of an async closure
                        this.with_lifetime_rib(
                            LifetimeRibKind::Elided(LifetimeRes::Infer),
                            |this| this.visit_expr(body),
                        );

                        debug!("(resolving function) leaving function");
                        this.in_func_body = previous_state;
                    }
                }
            })
        });
        self.diagnostic_metadata.current_function = previous_value;
    }
    fn visit_lifetime(&mut self, lifetime: &'ast Lifetime, use_ctxt: visit::LifetimeCtxt) {
        self.resolve_lifetime(lifetime, use_ctxt)
    }

    fn visit_generics(&mut self, generics: &'ast Generics) {
        self.visit_generic_params(
            &generics.params,
            self.diagnostic_metadata.current_self_item.is_some(),
        );
        for p in &generics.where_clause.predicates {
            self.visit_where_predicate(p);
        }
    }

    fn visit_closure_binder(&mut self, b: &'ast ClosureBinder) {
        match b {
            ClosureBinder::NotPresent => {}
            ClosureBinder::For { generic_params, .. } => {
                self.visit_generic_params(
                    &generic_params,
                    self.diagnostic_metadata.current_self_item.is_some(),
                );
            }
        }
    }

    fn visit_generic_arg(&mut self, arg: &'ast GenericArg) {
        debug!("visit_generic_arg({:?})", arg);
        let prev = replace(&mut self.diagnostic_metadata.currently_processing_generics, true);
        match arg {
            GenericArg::Type(ref ty) => {
                // We parse const arguments as path types as we cannot distinguish them during
                // parsing. We try to resolve that ambiguity by attempting resolution the type
                // namespace first, and if that fails we try again in the value namespace. If
                // resolution in the value namespace succeeds, we have an generic const argument on
                // our hands.
                if let TyKind::Path(ref qself, ref path) = ty.kind {
                    // We cannot disambiguate multi-segment paths right now as that requires type
                    // checking.
                    if path.segments.len() == 1 && path.segments[0].args.is_none() {
                        let mut check_ns = |ns| {
                            self.maybe_resolve_ident_in_lexical_scope(path.segments[0].ident, ns)
                                .is_some()
                        };
                        if !check_ns(TypeNS) && check_ns(ValueNS) {
                            // This must be equivalent to `visit_anon_const`, but we cannot call it
                            // directly due to visitor lifetimes so we have to copy-paste some code.
                            //
                            // Note that we might not be inside of an repeat expression here,
                            // but considering that `IsRepeatExpr` is only relevant for
                            // non-trivial constants this is doesn't matter.
                            self.with_constant_rib(
                                IsRepeatExpr::No,
                                ConstantHasGenerics::Yes,
                                None,
                                |this| {
                                    this.smart_resolve_path(
                                        ty.id,
                                        qself,
                                        path,
                                        PathSource::Expr(None),
                                    );

                                    if let Some(ref qself) = *qself {
                                        this.visit_ty(&qself.ty);
                                    }
                                    this.visit_path(path, ty.id);
                                },
                            );

                            self.diagnostic_metadata.currently_processing_generics = prev;
                            return;
                        }
                    }
                }

                self.visit_ty(ty);
            }
            GenericArg::Lifetime(lt) => self.visit_lifetime(lt, visit::LifetimeCtxt::GenericArg),
            GenericArg::Const(ct) => self.visit_anon_const(ct),
        }
        self.diagnostic_metadata.currently_processing_generics = prev;
    }

    fn visit_assoc_constraint(&mut self, constraint: &'ast AssocConstraint) {
        self.visit_ident(constraint.ident);
        if let Some(ref gen_args) = constraint.gen_args {
            // Forbid anonymous lifetimes in GAT parameters until proper semantics are decided.
            self.with_lifetime_rib(LifetimeRibKind::AnonymousReportError, |this| {
                this.visit_generic_args(gen_args)
            });
        }
        match constraint.kind {
            AssocConstraintKind::Equality { ref term } => match term {
                Term::Ty(ty) => self.visit_ty(ty),
                Term::Const(c) => self.visit_anon_const(c),
            },
            AssocConstraintKind::Bound { ref bounds } => {
                walk_list!(self, visit_param_bound, bounds, BoundKind::Bound);
            }
        }
    }

    fn visit_path_segment(&mut self, path_segment: &'ast PathSegment) {
        if let Some(ref args) = path_segment.args {
            match &**args {
                GenericArgs::AngleBracketed(..) => visit::walk_generic_args(self, args),
                GenericArgs::Parenthesized(p_args) => {
                    // Probe the lifetime ribs to know how to behave.
                    for rib in self.lifetime_ribs.iter().rev() {
                        match rib.kind {
                            // We are inside a `PolyTraitRef`. The lifetimes are
                            // to be intoduced in that (maybe implicit) `for<>` binder.
                            LifetimeRibKind::Generics {
                                binder,
                                kind: LifetimeBinderKind::PolyTrait,
                                ..
                            } => {
                                self.with_lifetime_rib(
                                    LifetimeRibKind::AnonymousCreateParameter {
                                        binder,
                                        report_in_path: false,
                                    },
                                    |this| {
                                        this.resolve_fn_signature(
                                            binder,
                                            false,
                                            p_args.inputs.iter().map(|ty| (None, &**ty)),
                                            &p_args.output,
                                        )
                                    },
                                );
                                break;
                            }
                            // We have nowhere to introduce generics. Code is malformed,
                            // so use regular lifetime resolution to avoid spurious errors.
                            LifetimeRibKind::Item | LifetimeRibKind::Generics { .. } => {
                                visit::walk_generic_args(self, args);
                                break;
                            }
                            LifetimeRibKind::AnonymousCreateParameter { .. }
                            | LifetimeRibKind::AnonymousReportError
                            | LifetimeRibKind::Elided(_)
                            | LifetimeRibKind::ElisionFailure
                            | LifetimeRibKind::AnonConst
                            | LifetimeRibKind::ConstGeneric => {}
                        }
                    }
                }
            }
        }
    }

    fn visit_where_predicate(&mut self, p: &'ast WherePredicate) {
        debug!("visit_where_predicate {:?}", p);
        let previous_value =
            replace(&mut self.diagnostic_metadata.current_where_predicate, Some(p));
        self.with_lifetime_rib(LifetimeRibKind::AnonymousReportError, |this| {
            if let WherePredicate::BoundPredicate(WhereBoundPredicate {
                ref bounded_ty,
                ref bounds,
                ref bound_generic_params,
                span: predicate_span,
                ..
            }) = p
            {
                let span = predicate_span.shrink_to_lo().to(bounded_ty.span.shrink_to_lo());
                this.with_generic_param_rib(
                    &bound_generic_params,
                    NormalRibKind,
                    LifetimeRibKind::Generics {
                        binder: bounded_ty.id,
                        kind: LifetimeBinderKind::WhereBound,
                        span,
                    },
                    |this| {
                        this.visit_generic_params(&bound_generic_params, false);
                        this.visit_ty(bounded_ty);
                        for bound in bounds {
                            this.visit_param_bound(bound, BoundKind::Bound)
                        }
                    },
                );
            } else {
                visit::walk_where_predicate(this, p);
            }
        });
        self.diagnostic_metadata.current_where_predicate = previous_value;
    }

    fn visit_inline_asm(&mut self, asm: &'ast InlineAsm) {
        for (op, _) in &asm.operands {
            match op {
                InlineAsmOperand::In { expr, .. }
                | InlineAsmOperand::Out { expr: Some(expr), .. }
                | InlineAsmOperand::InOut { expr, .. } => self.visit_expr(expr),
                InlineAsmOperand::Out { expr: None, .. } => {}
                InlineAsmOperand::SplitInOut { in_expr, out_expr, .. } => {
                    self.visit_expr(in_expr);
                    if let Some(out_expr) = out_expr {
                        self.visit_expr(out_expr);
                    }
                }
                InlineAsmOperand::Const { anon_const, .. } => {
                    // Although this is `DefKind::AnonConst`, it is allowed to reference outer
                    // generic parameters like an inline const.
                    self.resolve_inline_const(anon_const);
                }
                InlineAsmOperand::Sym { sym } => self.visit_inline_asm_sym(sym),
            }
        }
    }

    fn visit_inline_asm_sym(&mut self, sym: &'ast InlineAsmSym) {
        // This is similar to the code for AnonConst.
        self.with_rib(ValueNS, InlineAsmSymRibKind, |this| {
            this.with_rib(TypeNS, InlineAsmSymRibKind, |this| {
                this.with_label_rib(InlineAsmSymRibKind, |this| {
                    this.smart_resolve_path(sym.id, &sym.qself, &sym.path, PathSource::Expr(None));
                    visit::walk_inline_asm_sym(this, sym);
                });
            })
        });
    }

    fn visit_variant(&mut self, v: &'ast Variant) {
        self.resolve_doc_links(&v.attrs, MaybeExported::Ok(v.id));
        visit::walk_variant(self, v)
    }

    fn visit_field_def(&mut self, f: &'ast FieldDef) {
        self.resolve_doc_links(&f.attrs, MaybeExported::Ok(f.id));
        visit::walk_field_def(self, f)
    }
}

impl<'a: 'ast, 'b, 'ast, 'tcx> LateResolutionVisitor<'a, 'b, 'ast, 'tcx> {
    fn new(resolver: &'b mut Resolver<'a, 'tcx>) -> LateResolutionVisitor<'a, 'b, 'ast, 'tcx> {
        // During late resolution we only track the module component of the parent scope,
        // although it may be useful to track other components as well for diagnostics.
        let graph_root = resolver.graph_root;
        let parent_scope = ParentScope::module(graph_root, resolver);
        let start_rib_kind = ModuleRibKind(graph_root);
        LateResolutionVisitor {
            r: resolver,
            parent_scope,
            ribs: PerNS {
                value_ns: vec![Rib::new(start_rib_kind)],
                type_ns: vec![Rib::new(start_rib_kind)],
                macro_ns: vec![Rib::new(start_rib_kind)],
            },
            last_block_rib: None,
            label_ribs: Vec::new(),
            lifetime_ribs: Vec::new(),
            lifetime_elision_candidates: None,
            current_trait_ref: None,
            diagnostic_metadata: Box::new(DiagnosticMetadata::default()),
            // errors at module scope should always be reported
            in_func_body: false,
            lifetime_uses: Default::default(),
        }
    }

    fn maybe_resolve_ident_in_lexical_scope(
        &mut self,
        ident: Ident,
        ns: Namespace,
    ) -> Option<LexicalScopeBinding<'a>> {
        self.r.resolve_ident_in_lexical_scope(
            ident,
            ns,
            &self.parent_scope,
            None,
            &self.ribs[ns],
            None,
        )
    }

    fn resolve_ident_in_lexical_scope(
        &mut self,
        ident: Ident,
        ns: Namespace,
        finalize: Option<Finalize>,
        ignore_binding: Option<&'a NameBinding<'a>>,
    ) -> Option<LexicalScopeBinding<'a>> {
        self.r.resolve_ident_in_lexical_scope(
            ident,
            ns,
            &self.parent_scope,
            finalize,
            &self.ribs[ns],
            ignore_binding,
        )
    }

    fn resolve_path(
        &mut self,
        path: &[Segment],
        opt_ns: Option<Namespace>, // `None` indicates a module path in import
        finalize: Option<Finalize>,
    ) -> PathResult<'a> {
        self.r.resolve_path_with_ribs(
            path,
            opt_ns,
            &self.parent_scope,
            finalize,
            Some(&self.ribs),
            None,
        )
    }

    // AST resolution
    //
    // We maintain a list of value ribs and type ribs.
    //
    // Simultaneously, we keep track of the current position in the module
    // graph in the `parent_scope.module` pointer. When we go to resolve a name in
    // the value or type namespaces, we first look through all the ribs and
    // then query the module graph. When we resolve a name in the module
    // namespace, we can skip all the ribs (since nested modules are not
    // allowed within blocks in Rust) and jump straight to the current module
    // graph node.
    //
    // Named implementations are handled separately. When we find a method
    // call, we consult the module node to find all of the implementations in
    // scope. This information is lazily cached in the module node. We then
    // generate a fake "implementation scope" containing all the
    // implementations thus found, for compatibility with old resolve pass.

    /// Do some `work` within a new innermost rib of the given `kind` in the given namespace (`ns`).
    fn with_rib<T>(
        &mut self,
        ns: Namespace,
        kind: RibKind<'a>,
        work: impl FnOnce(&mut Self) -> T,
    ) -> T {
        self.ribs[ns].push(Rib::new(kind));
        let ret = work(self);
        self.ribs[ns].pop();
        ret
    }

    fn with_scope<T>(&mut self, id: NodeId, f: impl FnOnce(&mut Self) -> T) -> T {
        if let Some(module) = self.r.get_module(self.r.local_def_id(id).to_def_id()) {
            // Move down in the graph.
            let orig_module = replace(&mut self.parent_scope.module, module);
            self.with_rib(ValueNS, ModuleRibKind(module), |this| {
                this.with_rib(TypeNS, ModuleRibKind(module), |this| {
                    let ret = f(this);
                    this.parent_scope.module = orig_module;
                    ret
                })
            })
        } else {
            f(self)
        }
    }

    fn visit_generic_params(&mut self, params: &'ast [GenericParam], add_self_upper: bool) {
        // For type parameter defaults, we have to ban access
        // to following type parameters, as the InternalSubsts can only
        // provide previous type parameters as they're built. We
        // put all the parameters on the ban list and then remove
        // them one by one as they are processed and become available.
        let mut forward_ty_ban_rib = Rib::new(ForwardGenericParamBanRibKind);
        let mut forward_const_ban_rib = Rib::new(ForwardGenericParamBanRibKind);
        for param in params.iter() {
            match param.kind {
                GenericParamKind::Type { .. } => {
                    forward_ty_ban_rib
                        .bindings
                        .insert(Ident::with_dummy_span(param.ident.name), Res::Err);
                }
                GenericParamKind::Const { .. } => {
                    forward_const_ban_rib
                        .bindings
                        .insert(Ident::with_dummy_span(param.ident.name), Res::Err);
                }
                GenericParamKind::Lifetime => {}
            }
        }

        // rust-lang/rust#61631: The type `Self` is essentially
        // another type parameter. For ADTs, we consider it
        // well-defined only after all of the ADT type parameters have
        // been provided. Therefore, we do not allow use of `Self`
        // anywhere in ADT type parameter defaults.
        //
        // (We however cannot ban `Self` for defaults on *all* generic
        // lists; e.g. trait generics can usefully refer to `Self`,
        // such as in the case of `trait Add<Rhs = Self>`.)
        if add_self_upper {
            // (`Some` if + only if we are in ADT's generics.)
            forward_ty_ban_rib.bindings.insert(Ident::with_dummy_span(kw::SelfUpper), Res::Err);
        }

        self.with_lifetime_rib(LifetimeRibKind::AnonymousReportError, |this| {
            for param in params {
                match param.kind {
                    GenericParamKind::Lifetime => {
                        for bound in &param.bounds {
                            this.visit_param_bound(bound, BoundKind::Bound);
                        }
                    }
                    GenericParamKind::Type { ref default } => {
                        for bound in &param.bounds {
                            this.visit_param_bound(bound, BoundKind::Bound);
                        }

                        if let Some(ref ty) = default {
                            this.ribs[TypeNS].push(forward_ty_ban_rib);
                            this.ribs[ValueNS].push(forward_const_ban_rib);
                            this.visit_ty(ty);
                            forward_const_ban_rib = this.ribs[ValueNS].pop().unwrap();
                            forward_ty_ban_rib = this.ribs[TypeNS].pop().unwrap();
                        }

                        // Allow all following defaults to refer to this type parameter.
                        forward_ty_ban_rib
                            .bindings
                            .remove(&Ident::with_dummy_span(param.ident.name));
                    }
                    GenericParamKind::Const { ref ty, kw_span: _, ref default } => {
                        // Const parameters can't have param bounds.
                        assert!(param.bounds.is_empty());

                        this.ribs[TypeNS].push(Rib::new(ConstParamTyRibKind));
                        this.ribs[ValueNS].push(Rib::new(ConstParamTyRibKind));
                        this.with_lifetime_rib(LifetimeRibKind::ConstGeneric, |this| {
                            this.visit_ty(ty)
                        });
                        this.ribs[TypeNS].pop().unwrap();
                        this.ribs[ValueNS].pop().unwrap();

                        if let Some(ref expr) = default {
                            this.ribs[TypeNS].push(forward_ty_ban_rib);
                            this.ribs[ValueNS].push(forward_const_ban_rib);
                            this.with_lifetime_rib(LifetimeRibKind::ConstGeneric, |this| {
                                this.resolve_anon_const(expr, IsRepeatExpr::No)
                            });
                            forward_const_ban_rib = this.ribs[ValueNS].pop().unwrap();
                            forward_ty_ban_rib = this.ribs[TypeNS].pop().unwrap();
                        }

                        // Allow all following defaults to refer to this const parameter.
                        forward_const_ban_rib
                            .bindings
                            .remove(&Ident::with_dummy_span(param.ident.name));
                    }
                }
            }
        })
    }

    #[instrument(level = "debug", skip(self, work))]
    fn with_lifetime_rib<T>(
        &mut self,
        kind: LifetimeRibKind,
        work: impl FnOnce(&mut Self) -> T,
    ) -> T {
        self.lifetime_ribs.push(LifetimeRib::new(kind));
        let outer_elision_candidates = self.lifetime_elision_candidates.take();
        let ret = work(self);
        self.lifetime_elision_candidates = outer_elision_candidates;
        self.lifetime_ribs.pop();
        ret
    }

    #[instrument(level = "debug", skip(self))]
    fn resolve_lifetime(&mut self, lifetime: &'ast Lifetime, use_ctxt: visit::LifetimeCtxt) {
        let ident = lifetime.ident;

        if ident.name == kw::StaticLifetime {
            self.record_lifetime_res(
                lifetime.id,
                LifetimeRes::Static,
                LifetimeElisionCandidate::Named,
            );
            return;
        }

        if ident.name == kw::UnderscoreLifetime {
            return self.resolve_anonymous_lifetime(lifetime, false);
        }

        let mut lifetime_rib_iter = self.lifetime_ribs.iter().rev();
        while let Some(rib) = lifetime_rib_iter.next() {
            let normalized_ident = ident.normalize_to_macros_2_0();
            if let Some(&(_, res)) = rib.bindings.get(&normalized_ident) {
                self.record_lifetime_res(lifetime.id, res, LifetimeElisionCandidate::Named);

                if let LifetimeRes::Param { param, .. } = res {
                    match self.lifetime_uses.entry(param) {
                        Entry::Vacant(v) => {
                            debug!("First use of {:?} at {:?}", res, ident.span);
                            let use_set = self
                                .lifetime_ribs
                                .iter()
                                .rev()
                                .find_map(|rib| match rib.kind {
                                    // Do not suggest eliding a lifetime where an anonymous
                                    // lifetime would be illegal.
                                    LifetimeRibKind::Item
                                    | LifetimeRibKind::AnonymousReportError
                                    | LifetimeRibKind::ElisionFailure => Some(LifetimeUseSet::Many),
                                    // An anonymous lifetime is legal here, go ahead.
                                    LifetimeRibKind::AnonymousCreateParameter { .. } => {
                                        Some(LifetimeUseSet::One { use_span: ident.span, use_ctxt })
                                    }
                                    // Only report if eliding the lifetime would have the same
                                    // semantics.
                                    LifetimeRibKind::Elided(r) => Some(if res == r {
                                        LifetimeUseSet::One { use_span: ident.span, use_ctxt }
                                    } else {
                                        LifetimeUseSet::Many
                                    }),
                                    LifetimeRibKind::Generics { .. } => None,
                                    LifetimeRibKind::ConstGeneric | LifetimeRibKind::AnonConst => {
                                        span_bug!(ident.span, "unexpected rib kind: {:?}", rib.kind)
                                    }
                                })
                                .unwrap_or(LifetimeUseSet::Many);
                            debug!(?use_ctxt, ?use_set);
                            v.insert(use_set);
                        }
                        Entry::Occupied(mut o) => {
                            debug!("Many uses of {:?} at {:?}", res, ident.span);
                            *o.get_mut() = LifetimeUseSet::Many;
                        }
                    }
                }
                return;
            }

            match rib.kind {
                LifetimeRibKind::Item => break,
                LifetimeRibKind::ConstGeneric => {
                    self.emit_non_static_lt_in_const_generic_error(lifetime);
                    self.record_lifetime_res(
                        lifetime.id,
                        LifetimeRes::Error,
                        LifetimeElisionCandidate::Ignore,
                    );
                    return;
                }
                LifetimeRibKind::AnonConst => {
                    self.maybe_emit_forbidden_non_static_lifetime_error(lifetime);
                    self.record_lifetime_res(
                        lifetime.id,
                        LifetimeRes::Error,
                        LifetimeElisionCandidate::Ignore,
                    );
                    return;
                }
                LifetimeRibKind::AnonymousCreateParameter { .. }
                | LifetimeRibKind::Elided(_)
                | LifetimeRibKind::Generics { .. }
                | LifetimeRibKind::ElisionFailure
                | LifetimeRibKind::AnonymousReportError => {}
            }
        }

        let mut outer_res = None;
        for rib in lifetime_rib_iter {
            let normalized_ident = ident.normalize_to_macros_2_0();
            if let Some((&outer, _)) = rib.bindings.get_key_value(&normalized_ident) {
                outer_res = Some(outer);
                break;
            }
        }

        self.emit_undeclared_lifetime_error(lifetime, outer_res);
        self.record_lifetime_res(lifetime.id, LifetimeRes::Error, LifetimeElisionCandidate::Named);
    }

    #[instrument(level = "debug", skip(self))]
    fn resolve_anonymous_lifetime(&mut self, lifetime: &Lifetime, elided: bool) {
        debug_assert_eq!(lifetime.ident.name, kw::UnderscoreLifetime);

        let missing_lifetime = MissingLifetime {
            id: lifetime.id,
            span: lifetime.ident.span,
            kind: if elided {
                MissingLifetimeKind::Ampersand
            } else {
                MissingLifetimeKind::Underscore
            },
            count: 1,
        };
        let elision_candidate = LifetimeElisionCandidate::Missing(missing_lifetime);
        for (i, rib) in self.lifetime_ribs.iter().enumerate().rev() {
            debug!(?rib.kind);
            match rib.kind {
                LifetimeRibKind::AnonymousCreateParameter { binder, .. } => {
                    let res = self.create_fresh_lifetime(lifetime.id, lifetime.ident, binder);
                    self.record_lifetime_res(lifetime.id, res, elision_candidate);
                    return;
                }
                LifetimeRibKind::AnonymousReportError => {
                    let (msg, note) = if elided {
                        (
                            "`&` without an explicit lifetime name cannot be used here",
                            "explicit lifetime name needed here",
                        )
                    } else {
                        ("`'_` cannot be used here", "`'_` is a reserved lifetime name")
                    };
                    let mut diag = rustc_errors::struct_span_err!(
                        self.r.tcx.sess,
                        lifetime.ident.span,
                        E0637,
                        "{}",
                        msg,
                    );
                    diag.span_label(lifetime.ident.span, note);
                    if elided {
                        for rib in self.lifetime_ribs[i..].iter().rev() {
                            if let LifetimeRibKind::Generics {
                                span,
                                kind: LifetimeBinderKind::PolyTrait | LifetimeBinderKind::WhereBound,
                                ..
                            } = &rib.kind
                            {
                                diag.span_help(
                                    *span,
                                    "consider introducing a higher-ranked lifetime here with `for<'a>`",
                                );
                                break;
                            }
                        }
                    }
                    diag.emit();
                    self.record_lifetime_res(lifetime.id, LifetimeRes::Error, elision_candidate);
                    return;
                }
                LifetimeRibKind::Elided(res) => {
                    self.record_lifetime_res(lifetime.id, res, elision_candidate);
                    return;
                }
                LifetimeRibKind::ElisionFailure => {
                    self.diagnostic_metadata.current_elision_failures.push(missing_lifetime);
                    self.record_lifetime_res(lifetime.id, LifetimeRes::Error, elision_candidate);
                    return;
                }
                LifetimeRibKind::Item => break,
                LifetimeRibKind::Generics { .. } | LifetimeRibKind::ConstGeneric => {}
                LifetimeRibKind::AnonConst => {
                    // There is always an `Elided(LifetimeRes::Static)` inside an `AnonConst`.
                    span_bug!(lifetime.ident.span, "unexpected rib kind: {:?}", rib.kind)
                }
            }
        }
        self.record_lifetime_res(lifetime.id, LifetimeRes::Error, elision_candidate);
        self.report_missing_lifetime_specifiers(vec![missing_lifetime], None);
    }

    #[instrument(level = "debug", skip(self))]
    fn resolve_elided_lifetime(&mut self, anchor_id: NodeId, span: Span) {
        let id = self.r.next_node_id();
        let lt = Lifetime { id, ident: Ident::new(kw::UnderscoreLifetime, span) };

        self.record_lifetime_res(
            anchor_id,
            LifetimeRes::ElidedAnchor { start: id, end: NodeId::from_u32(id.as_u32() + 1) },
            LifetimeElisionCandidate::Ignore,
        );
        self.resolve_anonymous_lifetime(&lt, true);
    }

    #[instrument(level = "debug", skip(self))]
    fn create_fresh_lifetime(&mut self, id: NodeId, ident: Ident, binder: NodeId) -> LifetimeRes {
        debug_assert_eq!(ident.name, kw::UnderscoreLifetime);
        debug!(?ident.span);

        // Leave the responsibility to create the `LocalDefId` to lowering.
        let param = self.r.next_node_id();
        let res = LifetimeRes::Fresh { param, binder };

        // Record the created lifetime parameter so lowering can pick it up and add it to HIR.
        self.r
            .extra_lifetime_params_map
            .entry(binder)
            .or_insert_with(Vec::new)
            .push((ident, param, res));
        res
    }

    #[instrument(level = "debug", skip(self))]
    fn resolve_elided_lifetimes_in_path(
        &mut self,
        path_id: NodeId,
        partial_res: PartialRes,
        path: &[Segment],
        source: PathSource<'_>,
        path_span: Span,
    ) {
        let proj_start = path.len() - partial_res.unresolved_segments();
        for (i, segment) in path.iter().enumerate() {
            if segment.has_lifetime_args {
                continue;
            }
            let Some(segment_id) = segment.id else {
                continue;
            };

            // Figure out if this is a type/trait segment,
            // which may need lifetime elision performed.
            let type_def_id = match partial_res.base_res() {
                Res::Def(DefKind::AssocTy, def_id) if i + 2 == proj_start => {
                    self.r.tcx.parent(def_id)
                }
                Res::Def(DefKind::Variant, def_id) if i + 1 == proj_start => {
                    self.r.tcx.parent(def_id)
                }
                Res::Def(DefKind::Struct, def_id)
                | Res::Def(DefKind::Union, def_id)
                | Res::Def(DefKind::Enum, def_id)
                | Res::Def(DefKind::TyAlias, def_id)
                | Res::Def(DefKind::Trait, def_id)
                    if i + 1 == proj_start =>
                {
                    def_id
                }
                _ => continue,
            };

            let expected_lifetimes = self.r.item_generics_num_lifetimes(type_def_id);
            if expected_lifetimes == 0 {
                continue;
            }

            let node_ids = self.r.next_node_ids(expected_lifetimes);
            self.record_lifetime_res(
                segment_id,
                LifetimeRes::ElidedAnchor { start: node_ids.start, end: node_ids.end },
                LifetimeElisionCandidate::Ignore,
            );

            let inferred = match source {
                PathSource::Trait(..) | PathSource::TraitItem(..) | PathSource::Type => false,
                PathSource::Expr(..)
                | PathSource::Pat
                | PathSource::Struct
                | PathSource::TupleStruct(..) => true,
            };
            if inferred {
                // Do not create a parameter for patterns and expressions: type checking can infer
                // the appropriate lifetime for us.
                for id in node_ids {
                    self.record_lifetime_res(
                        id,
                        LifetimeRes::Infer,
                        LifetimeElisionCandidate::Named,
                    );
                }
                continue;
            }

            let elided_lifetime_span = if segment.has_generic_args {
                // If there are brackets, but not generic arguments, then use the opening bracket
                segment.args_span.with_hi(segment.args_span.lo() + BytePos(1))
            } else {
                // If there are no brackets, use the identifier span.
                // HACK: we use find_ancestor_inside to properly suggest elided spans in paths
                // originating from macros, since the segment's span might be from a macro arg.
                segment.ident.span.find_ancestor_inside(path_span).unwrap_or(path_span)
            };
            let ident = Ident::new(kw::UnderscoreLifetime, elided_lifetime_span);

            let missing_lifetime = MissingLifetime {
                id: node_ids.start,
                span: elided_lifetime_span,
                kind: if segment.has_generic_args {
                    MissingLifetimeKind::Comma
                } else {
                    MissingLifetimeKind::Brackets
                },
                count: expected_lifetimes,
            };
            let mut should_lint = true;
            for rib in self.lifetime_ribs.iter().rev() {
                match rib.kind {
                    // In create-parameter mode we error here because we don't want to support
                    // deprecated impl elision in new features like impl elision and `async fn`,
                    // both of which work using the `CreateParameter` mode:
                    //
                    //     impl Foo for std::cell::Ref<u32> // note lack of '_
                    //     async fn foo(_: std::cell::Ref<u32>) { ... }
                    LifetimeRibKind::AnonymousCreateParameter { report_in_path: true, .. } => {
                        let sess = self.r.tcx.sess;
                        let mut err = rustc_errors::struct_span_err!(
                            sess,
                            path_span,
                            E0726,
                            "implicit elided lifetime not allowed here"
                        );
                        rustc_errors::add_elided_lifetime_in_path_suggestion(
                            sess.source_map(),
                            &mut err,
                            expected_lifetimes,
                            path_span,
                            !segment.has_generic_args,
                            elided_lifetime_span,
                        );
                        err.emit();
                        should_lint = false;

                        for id in node_ids {
                            self.record_lifetime_res(
                                id,
                                LifetimeRes::Error,
                                LifetimeElisionCandidate::Named,
                            );
                        }
                        break;
                    }
                    // Do not create a parameter for patterns and expressions.
                    LifetimeRibKind::AnonymousCreateParameter { binder, .. } => {
                        // Group all suggestions into the first record.
                        let mut candidate = LifetimeElisionCandidate::Missing(missing_lifetime);
                        for id in node_ids {
                            let res = self.create_fresh_lifetime(id, ident, binder);
                            self.record_lifetime_res(
                                id,
                                res,
                                replace(&mut candidate, LifetimeElisionCandidate::Named),
                            );
                        }
                        break;
                    }
                    LifetimeRibKind::Elided(res) => {
                        let mut candidate = LifetimeElisionCandidate::Missing(missing_lifetime);
                        for id in node_ids {
                            self.record_lifetime_res(
                                id,
                                res,
                                replace(&mut candidate, LifetimeElisionCandidate::Ignore),
                            );
                        }
                        break;
                    }
                    LifetimeRibKind::ElisionFailure => {
                        self.diagnostic_metadata.current_elision_failures.push(missing_lifetime);
                        for id in node_ids {
                            self.record_lifetime_res(
                                id,
                                LifetimeRes::Error,
                                LifetimeElisionCandidate::Ignore,
                            );
                        }
                        break;
                    }
                    // `LifetimeRes::Error`, which would usually be used in the case of
                    // `ReportError`, is unsuitable here, as we don't emit an error yet. Instead,
                    // we simply resolve to an implicit lifetime, which will be checked later, at
                    // which point a suitable error will be emitted.
                    LifetimeRibKind::AnonymousReportError | LifetimeRibKind::Item => {
                        for id in node_ids {
                            self.record_lifetime_res(
                                id,
                                LifetimeRes::Error,
                                LifetimeElisionCandidate::Ignore,
                            );
                        }
                        self.report_missing_lifetime_specifiers(vec![missing_lifetime], None);
                        break;
                    }
                    LifetimeRibKind::Generics { .. } | LifetimeRibKind::ConstGeneric => {}
                    LifetimeRibKind::AnonConst => {
                        // There is always an `Elided(LifetimeRes::Static)` inside an `AnonConst`.
                        span_bug!(elided_lifetime_span, "unexpected rib kind: {:?}", rib.kind)
                    }
                }
            }

            if should_lint {
                self.r.lint_buffer.buffer_lint_with_diagnostic(
                    lint::builtin::ELIDED_LIFETIMES_IN_PATHS,
                    segment_id,
                    elided_lifetime_span,
                    "hidden lifetime parameters in types are deprecated",
                    lint::BuiltinLintDiagnostics::ElidedLifetimesInPaths(
                        expected_lifetimes,
                        path_span,
                        !segment.has_generic_args,
                        elided_lifetime_span,
                    ),
                );
            }
        }
    }

    #[instrument(level = "debug", skip(self))]
    fn record_lifetime_res(
        &mut self,
        id: NodeId,
        res: LifetimeRes,
        candidate: LifetimeElisionCandidate,
    ) {
        if let Some(prev_res) = self.r.lifetimes_res_map.insert(id, res) {
            panic!(
                "lifetime {:?} resolved multiple times ({:?} before, {:?} now)",
                id, prev_res, res
            )
        }
        match res {
            LifetimeRes::Param { .. } | LifetimeRes::Fresh { .. } | LifetimeRes::Static => {
                if let Some(ref mut candidates) = self.lifetime_elision_candidates {
                    candidates.push((res, candidate));
                }
            }
            LifetimeRes::Infer | LifetimeRes::Error | LifetimeRes::ElidedAnchor { .. } => {}
        }
    }

    #[instrument(level = "debug", skip(self))]
    fn record_lifetime_param(&mut self, id: NodeId, res: LifetimeRes) {
        if let Some(prev_res) = self.r.lifetimes_res_map.insert(id, res) {
            panic!(
                "lifetime parameter {:?} resolved multiple times ({:?} before, {:?} now)",
                id, prev_res, res
            )
        }
    }

    /// Perform resolution of a function signature, accounting for lifetime elision.
    #[instrument(level = "debug", skip(self, inputs))]
    fn resolve_fn_signature(
        &mut self,
        fn_id: NodeId,
        has_self: bool,
        inputs: impl Iterator<Item = (Option<&'ast Pat>, &'ast Ty)> + Clone,
        output_ty: &'ast FnRetTy,
    ) {
        // Add each argument to the rib.
        let elision_lifetime = self.resolve_fn_params(has_self, inputs);
        debug!(?elision_lifetime);

        let outer_failures = take(&mut self.diagnostic_metadata.current_elision_failures);
        let output_rib = if let Ok(res) = elision_lifetime.as_ref() {
            self.r.lifetime_elision_allowed.insert(fn_id);
            LifetimeRibKind::Elided(*res)
        } else {
            LifetimeRibKind::ElisionFailure
        };
        self.with_lifetime_rib(output_rib, |this| visit::walk_fn_ret_ty(this, &output_ty));
        let elision_failures =
            replace(&mut self.diagnostic_metadata.current_elision_failures, outer_failures);
        if !elision_failures.is_empty() {
            let Err(failure_info) = elision_lifetime else { bug!() };
            self.report_missing_lifetime_specifiers(elision_failures, Some(failure_info));
        }
    }

    /// Resolve inside function parameters and parameter types.
    /// Returns the lifetime for elision in fn return type,
    /// or diagnostic information in case of elision failure.
    fn resolve_fn_params(
        &mut self,
        has_self: bool,
        inputs: impl Iterator<Item = (Option<&'ast Pat>, &'ast Ty)>,
    ) -> Result<LifetimeRes, (Vec<MissingLifetime>, Vec<ElisionFnParameter>)> {
        enum Elision {
            /// We have not found any candidate.
            None,
            /// We have a candidate bound to `self`.
            Self_(LifetimeRes),
            /// We have a candidate bound to a parameter.
            Param(LifetimeRes),
            /// We failed elision.
            Err,
        }

        // Save elision state to reinstate it later.
        let outer_candidates = self.lifetime_elision_candidates.take();

        // Result of elision.
        let mut elision_lifetime = Elision::None;
        // Information for diagnostics.
        let mut parameter_info = Vec::new();
        let mut all_candidates = Vec::new();

        let mut bindings = smallvec![(PatBoundCtx::Product, Default::default())];
        for (index, (pat, ty)) in inputs.enumerate() {
            debug!(?pat, ?ty);
            self.with_lifetime_rib(LifetimeRibKind::Elided(LifetimeRes::Infer), |this| {
                if let Some(pat) = pat {
                    this.resolve_pattern(pat, PatternSource::FnParam, &mut bindings);
                }
            });

            // Record elision candidates only for this parameter.
            debug_assert_matches!(self.lifetime_elision_candidates, None);
            self.lifetime_elision_candidates = Some(Default::default());
            self.visit_ty(ty);
            let local_candidates = self.lifetime_elision_candidates.take();

            if let Some(candidates) = local_candidates {
                let distinct: FxHashSet<_> = candidates.iter().map(|(res, _)| *res).collect();
                let lifetime_count = distinct.len();
                if lifetime_count != 0 {
                    parameter_info.push(ElisionFnParameter {
                        index,
                        ident: if let Some(pat) = pat && let PatKind::Ident(_, ident, _) = pat.kind {
                            Some(ident)
                        } else {
                            None
                        },
                        lifetime_count,
                        span: ty.span,
                    });
                    all_candidates.extend(candidates.into_iter().filter_map(|(_, candidate)| {
                        match candidate {
                            LifetimeElisionCandidate::Ignore | LifetimeElisionCandidate::Named => {
                                None
                            }
                            LifetimeElisionCandidate::Missing(missing) => Some(missing),
                        }
                    }));
                }
                let mut distinct_iter = distinct.into_iter();
                if let Some(res) = distinct_iter.next() {
                    match elision_lifetime {
                        // We are the first parameter to bind lifetimes.
                        Elision::None => {
                            if distinct_iter.next().is_none() {
                                // We have a single lifetime => success.
                                elision_lifetime = Elision::Param(res)
                            } else {
                                // We have multiple lifetimes => error.
                                elision_lifetime = Elision::Err;
                            }
                        }
                        // We have 2 parameters that bind lifetimes => error.
                        Elision::Param(_) => elision_lifetime = Elision::Err,
                        // `self` elision takes precedence over everything else.
                        Elision::Self_(_) | Elision::Err => {}
                    }
                }
            }

            // Handle `self` specially.
            if index == 0 && has_self {
                let self_lifetime = self.find_lifetime_for_self(ty);
                if let Set1::One(lifetime) = self_lifetime {
                    // We found `self` elision.
                    elision_lifetime = Elision::Self_(lifetime);
                } else {
                    // We do not have `self` elision: disregard the `Elision::Param` that we may
                    // have found.
                    elision_lifetime = Elision::None;
                }
            }
            debug!("(resolving function / closure) recorded parameter");
        }

        // Reinstate elision state.
        debug_assert_matches!(self.lifetime_elision_candidates, None);
        self.lifetime_elision_candidates = outer_candidates;

        if let Elision::Param(res) | Elision::Self_(res) = elision_lifetime {
            return Ok(res);
        }

        // We do not have a candidate.
        Err((all_candidates, parameter_info))
    }

    /// List all the lifetimes that appear in the provided type.
    fn find_lifetime_for_self(&self, ty: &'ast Ty) -> Set1<LifetimeRes> {
        struct SelfVisitor<'r, 'a, 'tcx> {
            r: &'r Resolver<'a, 'tcx>,
            impl_self: Option<Res>,
            lifetime: Set1<LifetimeRes>,
        }

        impl SelfVisitor<'_, '_, '_> {
            // Look for `self: &'a Self` - also desugared from `&'a self`,
            // and if that matches, use it for elision and return early.
            fn is_self_ty(&self, ty: &Ty) -> bool {
                match ty.kind {
                    TyKind::ImplicitSelf => true,
                    TyKind::Path(None, _) => {
                        let path_res = self.r.partial_res_map[&ty.id].full_res();
                        if let Some(Res::SelfTyParam { .. } | Res::SelfTyAlias { .. }) = path_res {
                            return true;
                        }
                        self.impl_self.is_some() && path_res == self.impl_self
                    }
                    _ => false,
                }
            }
        }

        impl<'a> Visitor<'a> for SelfVisitor<'_, '_, '_> {
            fn visit_ty(&mut self, ty: &'a Ty) {
                trace!("SelfVisitor considering ty={:?}", ty);
                if let TyKind::Ref(lt, ref mt) = ty.kind && self.is_self_ty(&mt.ty) {
                    let lt_id = if let Some(lt) = lt {
                        lt.id
                    } else {
                        let res = self.r.lifetimes_res_map[&ty.id];
                        let LifetimeRes::ElidedAnchor { start, .. } = res else { bug!() };
                        start
                    };
                    let lt_res = self.r.lifetimes_res_map[&lt_id];
                    trace!("SelfVisitor inserting res={:?}", lt_res);
                    self.lifetime.insert(lt_res);
                }
                visit::walk_ty(self, ty)
            }
        }

        let impl_self = self
            .diagnostic_metadata
            .current_self_type
            .as_ref()
            .and_then(|ty| {
                if let TyKind::Path(None, _) = ty.kind {
                    self.r.partial_res_map.get(&ty.id)
                } else {
                    None
                }
            })
            .and_then(|res| res.full_res())
            .filter(|res| {
                // Permit the types that unambiguously always
                // result in the same type constructor being used
                // (it can't differ between `Self` and `self`).
                matches!(
                    res,
                    Res::Def(DefKind::Struct | DefKind::Union | DefKind::Enum, _,) | Res::PrimTy(_)
                )
            });
        let mut visitor = SelfVisitor { r: self.r, impl_self, lifetime: Set1::Empty };
        visitor.visit_ty(ty);
        trace!("SelfVisitor found={:?}", visitor.lifetime);
        visitor.lifetime
    }

    /// Searches the current set of local scopes for labels. Returns the `NodeId` of the resolved
    /// label and reports an error if the label is not found or is unreachable.
    fn resolve_label(&mut self, mut label: Ident) -> Result<(NodeId, Span), ResolutionError<'a>> {
        let mut suggestion = None;

        for i in (0..self.label_ribs.len()).rev() {
            let rib = &self.label_ribs[i];

            if let MacroDefinition(def) = rib.kind {
                // If an invocation of this macro created `ident`, give up on `ident`
                // and switch to `ident`'s source from the macro definition.
                if def == self.r.macro_def(label.span.ctxt()) {
                    label.span.remove_mark();
                }
            }

            let ident = label.normalize_to_macro_rules();
            if let Some((ident, id)) = rib.bindings.get_key_value(&ident) {
                let definition_span = ident.span;
                return if self.is_label_valid_from_rib(i) {
                    Ok((*id, definition_span))
                } else {
                    Err(ResolutionError::UnreachableLabel {
                        name: label.name,
                        definition_span,
                        suggestion,
                    })
                };
            }

            // Diagnostics: Check if this rib contains a label with a similar name, keep track of
            // the first such label that is encountered.
            suggestion = suggestion.or_else(|| self.suggestion_for_label_in_rib(i, label));
        }

        Err(ResolutionError::UndeclaredLabel { name: label.name, suggestion })
    }

    /// Determine whether or not a label from the `rib_index`th label rib is reachable.
    fn is_label_valid_from_rib(&self, rib_index: usize) -> bool {
        let ribs = &self.label_ribs[rib_index + 1..];

        for rib in ribs {
            if rib.kind.is_label_barrier() {
                return false;
            }
        }

        true
    }

    fn resolve_adt(&mut self, item: &'ast Item, generics: &'ast Generics) {
        debug!("resolve_adt");
        self.with_current_self_item(item, |this| {
            this.with_generic_param_rib(
                &generics.params,
                ItemRibKind(HasGenericParams::Yes(generics.span)),
                LifetimeRibKind::Generics {
                    binder: item.id,
                    kind: LifetimeBinderKind::Item,
                    span: generics.span,
                },
                |this| {
                    let item_def_id = this.r.local_def_id(item.id).to_def_id();
                    this.with_self_rib(
                        Res::SelfTyAlias {
                            alias_to: item_def_id,
                            forbid_generic: false,
                            is_trait_impl: false,
                        },
                        |this| {
                            visit::walk_item(this, item);
                        },
                    );
                },
            );
        });
    }

    fn future_proof_import(&mut self, use_tree: &UseTree) {
        let segments = &use_tree.prefix.segments;
        if !segments.is_empty() {
            let ident = segments[0].ident;
            if ident.is_path_segment_keyword() || ident.span.is_rust_2015() {
                return;
            }

            let nss = match use_tree.kind {
                UseTreeKind::Simple(..) if segments.len() == 1 => &[TypeNS, ValueNS][..],
                _ => &[TypeNS],
            };
            let report_error = |this: &Self, ns| {
                let what = if ns == TypeNS { "type parameters" } else { "local variables" };
                if this.should_report_errs() {
                    this.r
                        .tcx
                        .sess
                        .span_err(ident.span, &format!("imports cannot refer to {}", what));
                }
            };

            for &ns in nss {
                match self.maybe_resolve_ident_in_lexical_scope(ident, ns) {
                    Some(LexicalScopeBinding::Res(..)) => {
                        report_error(self, ns);
                    }
                    Some(LexicalScopeBinding::Item(binding)) => {
                        if let Some(LexicalScopeBinding::Res(..)) =
                            self.resolve_ident_in_lexical_scope(ident, ns, None, Some(binding))
                        {
                            report_error(self, ns);
                        }
                    }
                    None => {}
                }
            }
        } else if let UseTreeKind::Nested(use_trees) = &use_tree.kind {
            for (use_tree, _) in use_trees {
                self.future_proof_import(use_tree);
            }
        }
    }

    fn resolve_item(&mut self, item: &'ast Item) {
        let mod_inner_docs =
            matches!(item.kind, ItemKind::Mod(..)) && rustdoc::inner_docs(&item.attrs);
        if !mod_inner_docs && !matches!(item.kind, ItemKind::Impl(..)) {
            self.resolve_doc_links(&item.attrs, MaybeExported::Ok(item.id));
        }

        let name = item.ident.name;
        debug!("(resolving item) resolving {} ({:?})", name, item.kind);

        match item.kind {
            ItemKind::TyAlias(box TyAlias { ref generics, .. }) => {
                self.with_generic_param_rib(
                    &generics.params,
                    ItemRibKind(HasGenericParams::Yes(generics.span)),
                    LifetimeRibKind::Generics {
                        binder: item.id,
                        kind: LifetimeBinderKind::Item,
                        span: generics.span,
                    },
                    |this| visit::walk_item(this, item),
                );
            }

            ItemKind::Fn(box Fn { ref generics, .. }) => {
                self.with_generic_param_rib(
                    &generics.params,
                    ItemRibKind(HasGenericParams::Yes(generics.span)),
                    LifetimeRibKind::Generics {
                        binder: item.id,
                        kind: LifetimeBinderKind::Function,
                        span: generics.span,
                    },
                    |this| visit::walk_item(this, item),
                );
            }

            ItemKind::Enum(_, ref generics)
            | ItemKind::Struct(_, ref generics)
            | ItemKind::Union(_, ref generics) => {
                self.resolve_adt(item, generics);
            }

            ItemKind::Impl(box Impl {
                ref generics,
                ref of_trait,
                ref self_ty,
                items: ref impl_items,
                ..
            }) => {
                self.diagnostic_metadata.current_impl_items = Some(impl_items);
                self.resolve_implementation(
                    &item.attrs,
                    generics,
                    of_trait,
                    &self_ty,
                    item.id,
                    impl_items,
                );
                self.diagnostic_metadata.current_impl_items = None;
            }

            ItemKind::Trait(box Trait { ref generics, ref bounds, ref items, .. }) => {
                // Create a new rib for the trait-wide type parameters.
                self.with_generic_param_rib(
                    &generics.params,
                    ItemRibKind(HasGenericParams::Yes(generics.span)),
                    LifetimeRibKind::Generics {
                        binder: item.id,
                        kind: LifetimeBinderKind::Item,
                        span: generics.span,
                    },
                    |this| {
                        let local_def_id = this.r.local_def_id(item.id).to_def_id();
                        this.with_self_rib(Res::SelfTyParam { trait_: local_def_id }, |this| {
                            this.visit_generics(generics);
                            walk_list!(this, visit_param_bound, bounds, BoundKind::SuperTraits);
                            this.resolve_trait_items(items);
                        });
                    },
                );
            }

            ItemKind::TraitAlias(ref generics, ref bounds) => {
                // Create a new rib for the trait-wide type parameters.
                self.with_generic_param_rib(
                    &generics.params,
                    ItemRibKind(HasGenericParams::Yes(generics.span)),
                    LifetimeRibKind::Generics {
                        binder: item.id,
                        kind: LifetimeBinderKind::Item,
                        span: generics.span,
                    },
                    |this| {
                        let local_def_id = this.r.local_def_id(item.id).to_def_id();
                        this.with_self_rib(Res::SelfTyParam { trait_: local_def_id }, |this| {
                            this.visit_generics(generics);
                            walk_list!(this, visit_param_bound, bounds, BoundKind::Bound);
                        });
                    },
                );
            }

            ItemKind::Mod(..) => {
                self.with_scope(item.id, |this| {
                    if mod_inner_docs {
                        this.resolve_doc_links(&item.attrs, MaybeExported::Ok(item.id));
                    }
                    let old_macro_rules = this.parent_scope.macro_rules;
                    visit::walk_item(this, item);
                    // Maintain macro_rules scopes in the same way as during early resolution
                    // for diagnostics and doc links.
                    if item.attrs.iter().all(|attr| {
                        !attr.has_name(sym::macro_use) && !attr.has_name(sym::macro_escape)
                    }) {
                        this.parent_scope.macro_rules = old_macro_rules;
                    }
                });
            }

            ItemKind::Static(ref ty, _, ref expr) | ItemKind::Const(_, ref ty, ref expr) => {
                self.with_static_rib(|this| {
                    this.with_lifetime_rib(LifetimeRibKind::Elided(LifetimeRes::Static), |this| {
                        this.visit_ty(ty);
                    });
                    this.with_lifetime_rib(LifetimeRibKind::Elided(LifetimeRes::Infer), |this| {
                        if let Some(expr) = expr {
                            let constant_item_kind = match item.kind {
                                ItemKind::Const(..) => ConstantItemKind::Const,
                                ItemKind::Static(..) => ConstantItemKind::Static,
                                _ => unreachable!(),
                            };
                            // We already forbid generic params because of the above item rib,
                            // so it doesn't matter whether this is a trivial constant.
                            this.with_constant_rib(
                                IsRepeatExpr::No,
                                ConstantHasGenerics::Yes,
                                Some((item.ident, constant_item_kind)),
                                |this| this.visit_expr(expr),
                            );
                        }
                    });
                });
            }

            ItemKind::Use(ref use_tree) => {
                self.future_proof_import(use_tree);
            }

            ItemKind::MacroDef(ref macro_def) => {
                // Maintain macro_rules scopes in the same way as during early resolution
                // for diagnostics and doc links.
                if macro_def.macro_rules {
                    let def_id = self.r.local_def_id(item.id);
                    self.parent_scope.macro_rules = self.r.macro_rules_scopes[&def_id];
                }
            }

            ItemKind::ForeignMod(_) | ItemKind::GlobalAsm(_) => {
                visit::walk_item(self, item);
            }

            ItemKind::ExternCrate(..) => {}

            ItemKind::MacCall(_) => panic!("unexpanded macro in resolve!"),
        }
    }

    fn with_generic_param_rib<'c, F>(
        &'c mut self,
        params: &'c [GenericParam],
        kind: RibKind<'a>,
        lifetime_kind: LifetimeRibKind,
        f: F,
    ) where
        F: FnOnce(&mut Self),
    {
        debug!("with_generic_param_rib");
        let LifetimeRibKind::Generics { binder, span: generics_span, kind: generics_kind, .. }
            = lifetime_kind else { panic!() };

        let mut function_type_rib = Rib::new(kind);
        let mut function_value_rib = Rib::new(kind);
        let mut function_lifetime_rib = LifetimeRib::new(lifetime_kind);
        let mut seen_bindings = FxHashMap::default();
        // Store all seen lifetimes names from outer scopes.
        let mut seen_lifetimes = FxHashSet::default();

        // We also can't shadow bindings from the parent item
        if let AssocItemRibKind = kind {
            let mut add_bindings_for_ns = |ns| {
                let parent_rib = self.ribs[ns]
                    .iter()
                    .rfind(|r| matches!(r.kind, ItemRibKind(_)))
                    .expect("associated item outside of an item");
                seen_bindings
                    .extend(parent_rib.bindings.iter().map(|(ident, _)| (*ident, ident.span)));
            };
            add_bindings_for_ns(ValueNS);
            add_bindings_for_ns(TypeNS);
        }

        // Forbid shadowing lifetime bindings
        for rib in self.lifetime_ribs.iter().rev() {
            seen_lifetimes.extend(rib.bindings.iter().map(|(ident, _)| *ident));
            if let LifetimeRibKind::Item = rib.kind {
                break;
            }
        }

        for param in params {
            let ident = param.ident.normalize_to_macros_2_0();
            debug!("with_generic_param_rib: {}", param.id);

            if let GenericParamKind::Lifetime = param.kind
                && let Some(&original) = seen_lifetimes.get(&ident)
            {
                diagnostics::signal_lifetime_shadowing(self.r.tcx.sess, original, param.ident);
                // Record lifetime res, so lowering knows there is something fishy.
                self.record_lifetime_param(param.id, LifetimeRes::Error);
                continue;
            }

            match seen_bindings.entry(ident) {
                Entry::Occupied(entry) => {
                    let span = *entry.get();
                    let err = ResolutionError::NameAlreadyUsedInParameterList(ident.name, span);
                    self.report_error(param.ident.span, err);
                    if let GenericParamKind::Lifetime = param.kind {
                        // Record lifetime res, so lowering knows there is something fishy.
                        self.record_lifetime_param(param.id, LifetimeRes::Error);
                    }
                    continue;
                }
                Entry::Vacant(entry) => {
                    entry.insert(param.ident.span);
                }
            }

            if param.ident.name == kw::UnderscoreLifetime {
                rustc_errors::struct_span_err!(
                    self.r.tcx.sess,
                    param.ident.span,
                    E0637,
                    "`'_` cannot be used here"
                )
                .span_label(param.ident.span, "`'_` is a reserved lifetime name")
                .emit();
                // Record lifetime res, so lowering knows there is something fishy.
                self.record_lifetime_param(param.id, LifetimeRes::Error);
                continue;
            }

            if param.ident.name == kw::StaticLifetime {
                rustc_errors::struct_span_err!(
                    self.r.tcx.sess,
                    param.ident.span,
                    E0262,
                    "invalid lifetime parameter name: `{}`",
                    param.ident,
                )
                .span_label(param.ident.span, "'static is a reserved lifetime name")
                .emit();
                // Record lifetime res, so lowering knows there is something fishy.
                self.record_lifetime_param(param.id, LifetimeRes::Error);
                continue;
            }

            let def_id = self.r.local_def_id(param.id);

            // Plain insert (no renaming).
            let (rib, def_kind) = match param.kind {
                GenericParamKind::Type { .. } => (&mut function_type_rib, DefKind::TyParam),
                GenericParamKind::Const { .. } => (&mut function_value_rib, DefKind::ConstParam),
                GenericParamKind::Lifetime => {
                    let res = LifetimeRes::Param { param: def_id, binder };
                    self.record_lifetime_param(param.id, res);
                    function_lifetime_rib.bindings.insert(ident, (param.id, res));
                    continue;
                }
            };

            let res = match kind {
                ItemRibKind(..) | AssocItemRibKind => Res::Def(def_kind, def_id.to_def_id()),
                NormalRibKind => {
                    if self.r.tcx.sess.features_untracked().non_lifetime_binders {
                        Res::Def(def_kind, def_id.to_def_id())
                    } else {
                        Res::Err
                    }
                }
                _ => span_bug!(param.ident.span, "Unexpected rib kind {:?}", kind),
            };
            self.r.record_partial_res(param.id, PartialRes::new(res));
            rib.bindings.insert(ident, res);
        }

        self.lifetime_ribs.push(function_lifetime_rib);
        self.ribs[ValueNS].push(function_value_rib);
        self.ribs[TypeNS].push(function_type_rib);

        f(self);

        self.ribs[TypeNS].pop();
        self.ribs[ValueNS].pop();
        let function_lifetime_rib = self.lifetime_ribs.pop().unwrap();

        // Do not account for the parameters we just bound for function lifetime elision.
        if let Some(ref mut candidates) = self.lifetime_elision_candidates {
            for (_, res) in function_lifetime_rib.bindings.values() {
                candidates.retain(|(r, _)| r != res);
            }
        }

        if let LifetimeBinderKind::BareFnType
        | LifetimeBinderKind::WhereBound
        | LifetimeBinderKind::Function
        | LifetimeBinderKind::ImplBlock = generics_kind
        {
            self.maybe_report_lifetime_uses(generics_span, params)
        }
    }

    fn with_label_rib(&mut self, kind: RibKind<'a>, f: impl FnOnce(&mut Self)) {
        self.label_ribs.push(Rib::new(kind));
        f(self);
        self.label_ribs.pop();
    }

    fn with_static_rib(&mut self, f: impl FnOnce(&mut Self)) {
        let kind = ItemRibKind(HasGenericParams::No);
        self.with_rib(ValueNS, kind, |this| this.with_rib(TypeNS, kind, f))
    }

    // HACK(min_const_generics,const_evaluatable_unchecked): We
    // want to keep allowing `[0; std::mem::size_of::<*mut T>()]`
    // with a future compat lint for now. We do this by adding an
    // additional special case for repeat expressions.
    //
    // Note that we intentionally still forbid `[0; N + 1]` during
    // name resolution so that we don't extend the future
    // compat lint to new cases.
    #[instrument(level = "debug", skip(self, f))]
    fn with_constant_rib(
        &mut self,
        is_repeat: IsRepeatExpr,
        may_use_generics: ConstantHasGenerics,
        item: Option<(Ident, ConstantItemKind)>,
        f: impl FnOnce(&mut Self),
    ) {
        self.with_rib(ValueNS, ConstantItemRibKind(may_use_generics, item), |this| {
            this.with_rib(
                TypeNS,
                ConstantItemRibKind(
                    may_use_generics.force_yes_if(is_repeat == IsRepeatExpr::Yes),
                    item,
                ),
                |this| {
                    this.with_label_rib(ConstantItemRibKind(may_use_generics, item), f);
                },
            )
        });
    }

    fn with_current_self_type<T>(&mut self, self_type: &Ty, f: impl FnOnce(&mut Self) -> T) -> T {
        // Handle nested impls (inside fn bodies)
        let previous_value =
            replace(&mut self.diagnostic_metadata.current_self_type, Some(self_type.clone()));
        let result = f(self);
        self.diagnostic_metadata.current_self_type = previous_value;
        result
    }

    fn with_current_self_item<T>(&mut self, self_item: &Item, f: impl FnOnce(&mut Self) -> T) -> T {
        let previous_value =
            replace(&mut self.diagnostic_metadata.current_self_item, Some(self_item.id));
        let result = f(self);
        self.diagnostic_metadata.current_self_item = previous_value;
        result
    }

    /// When evaluating a `trait` use its associated types' idents for suggestions in E0412.
    fn resolve_trait_items(&mut self, trait_items: &'ast [P<AssocItem>]) {
        let trait_assoc_items =
            replace(&mut self.diagnostic_metadata.current_trait_assoc_items, Some(&trait_items));

        let walk_assoc_item =
            |this: &mut Self, generics: &Generics, kind, item: &'ast AssocItem| {
                this.with_generic_param_rib(
                    &generics.params,
                    AssocItemRibKind,
                    LifetimeRibKind::Generics { binder: item.id, span: generics.span, kind },
                    |this| visit::walk_assoc_item(this, item, AssocCtxt::Trait),
                );
            };

        for item in trait_items {
            self.resolve_doc_links(&item.attrs, MaybeExported::Ok(item.id));
            match &item.kind {
                AssocItemKind::Const(_, ty, default) => {
                    self.visit_ty(ty);
                    // Only impose the restrictions of `ConstRibKind` for an
                    // actual constant expression in a provided default.
                    if let Some(expr) = default {
                        // We allow arbitrary const expressions inside of associated consts,
                        // even if they are potentially not const evaluatable.
                        //
                        // Type parameters can already be used and as associated consts are
                        // not used as part of the type system, this is far less surprising.
                        self.with_lifetime_rib(
                            LifetimeRibKind::Elided(LifetimeRes::Infer),
                            |this| {
                                this.with_constant_rib(
                                    IsRepeatExpr::No,
                                    ConstantHasGenerics::Yes,
                                    None,
                                    |this| this.visit_expr(expr),
                                )
                            },
                        );
                    }
                }
                AssocItemKind::Fn(box Fn { generics, .. }) => {
                    walk_assoc_item(self, generics, LifetimeBinderKind::Function, item);
                }
                AssocItemKind::Type(box TyAlias { generics, .. }) => self
                    .with_lifetime_rib(LifetimeRibKind::AnonymousReportError, |this| {
                        walk_assoc_item(this, generics, LifetimeBinderKind::Item, item)
                    }),
                AssocItemKind::MacCall(_) => {
                    panic!("unexpanded macro in resolve!")
                }
            };
        }

        self.diagnostic_metadata.current_trait_assoc_items = trait_assoc_items;
    }

    /// This is called to resolve a trait reference from an `impl` (i.e., `impl Trait for Foo`).
    fn with_optional_trait_ref<T>(
        &mut self,
        opt_trait_ref: Option<&TraitRef>,
        self_type: &'ast Ty,
        f: impl FnOnce(&mut Self, Option<DefId>) -> T,
    ) -> T {
        let mut new_val = None;
        let mut new_id = None;
        if let Some(trait_ref) = opt_trait_ref {
            let path: Vec<_> = Segment::from_path(&trait_ref.path);
            self.diagnostic_metadata.currently_processing_impl_trait =
                Some((trait_ref.clone(), self_type.clone()));
            let res = self.smart_resolve_path_fragment(
                &None,
                &path,
                PathSource::Trait(AliasPossibility::No),
                Finalize::new(trait_ref.ref_id, trait_ref.path.span),
            );
            self.diagnostic_metadata.currently_processing_impl_trait = None;
            if let Some(def_id) = res.expect_full_res().opt_def_id() {
                new_id = Some(def_id);
                new_val = Some((self.r.expect_module(def_id), trait_ref.clone()));
            }
        }
        let original_trait_ref = replace(&mut self.current_trait_ref, new_val);
        let result = f(self, new_id);
        self.current_trait_ref = original_trait_ref;
        result
    }

    fn with_self_rib_ns(&mut self, ns: Namespace, self_res: Res, f: impl FnOnce(&mut Self)) {
        let mut self_type_rib = Rib::new(NormalRibKind);

        // Plain insert (no renaming, since types are not currently hygienic)
        self_type_rib.bindings.insert(Ident::with_dummy_span(kw::SelfUpper), self_res);
        self.ribs[ns].push(self_type_rib);
        f(self);
        self.ribs[ns].pop();
    }

    fn with_self_rib(&mut self, self_res: Res, f: impl FnOnce(&mut Self)) {
        self.with_self_rib_ns(TypeNS, self_res, f)
    }

    fn resolve_implementation(
        &mut self,
        attrs: &[ast::Attribute],
        generics: &'ast Generics,
        opt_trait_reference: &'ast Option<TraitRef>,
        self_type: &'ast Ty,
        item_id: NodeId,
        impl_items: &'ast [P<AssocItem>],
    ) {
        debug!("resolve_implementation");
        // If applicable, create a rib for the type parameters.
        self.with_generic_param_rib(
            &generics.params,
            ItemRibKind(HasGenericParams::Yes(generics.span)),
            LifetimeRibKind::Generics {
                span: generics.span,
                binder: item_id,
                kind: LifetimeBinderKind::ImplBlock,
            },
            |this| {
                // Dummy self type for better errors if `Self` is used in the trait path.
                this.with_self_rib(Res::SelfTyParam { trait_: LOCAL_CRATE.as_def_id() }, |this| {
                    this.with_lifetime_rib(
                        LifetimeRibKind::AnonymousCreateParameter {
                            binder: item_id,
                            report_in_path: true
                        },
                        |this| {
                            // Resolve the trait reference, if necessary.
                            this.with_optional_trait_ref(
                                opt_trait_reference.as_ref(),
                                self_type,
                                |this, trait_id| {
                                    this.resolve_doc_links(attrs, MaybeExported::Impl(trait_id));

                                    let item_def_id = this.r.local_def_id(item_id);

                                    // Register the trait definitions from here.
                                    if let Some(trait_id) = trait_id {
                                        this.r
                                            .trait_impls
                                            .entry(trait_id)
                                            .or_default()
                                            .push(item_def_id);
                                    }

                                    let item_def_id = item_def_id.to_def_id();
                                    let res = Res::SelfTyAlias {
                                        alias_to: item_def_id,
                                        forbid_generic: false,
                                        is_trait_impl: trait_id.is_some()
                                    };
                                    this.with_self_rib(res, |this| {
                                        if let Some(trait_ref) = opt_trait_reference.as_ref() {
                                            // Resolve type arguments in the trait path.
                                            visit::walk_trait_ref(this, trait_ref);
                                        }
                                        // Resolve the self type.
                                        this.visit_ty(self_type);
                                        // Resolve the generic parameters.
                                        this.visit_generics(generics);

                                        // Resolve the items within the impl.
                                        this.with_current_self_type(self_type, |this| {
                                            this.with_self_rib_ns(ValueNS, Res::SelfCtor(item_def_id), |this| {
                                                debug!("resolve_implementation with_self_rib_ns(ValueNS, ...)");
                                                let mut seen_trait_items = Default::default();
                                                for item in impl_items {
                                                    this.resolve_impl_item(&**item, &mut seen_trait_items, trait_id);
                                                }
                                            });
                                        });
                                    });
                                },
                            )
                        },
                    );
                });
            },
        );
    }

    fn resolve_impl_item(
        &mut self,
        item: &'ast AssocItem,
        seen_trait_items: &mut FxHashMap<DefId, Span>,
        trait_id: Option<DefId>,
    ) {
        use crate::ResolutionError::*;
        self.resolve_doc_links(&item.attrs, MaybeExported::ImplItem(trait_id.ok_or(&item.vis)));
        match &item.kind {
            AssocItemKind::Const(_, ty, default) => {
                debug!("resolve_implementation AssocItemKind::Const");
                // If this is a trait impl, ensure the const
                // exists in trait
                self.check_trait_item(
                    item.id,
                    item.ident,
                    &item.kind,
                    ValueNS,
                    item.span,
                    seen_trait_items,
                    |i, s, c| ConstNotMemberOfTrait(i, s, c),
                );

                self.visit_ty(ty);
                if let Some(expr) = default {
                    // We allow arbitrary const expressions inside of associated consts,
                    // even if they are potentially not const evaluatable.
                    //
                    // Type parameters can already be used and as associated consts are
                    // not used as part of the type system, this is far less surprising.
                    self.with_lifetime_rib(LifetimeRibKind::Elided(LifetimeRes::Infer), |this| {
                        this.with_constant_rib(
                            IsRepeatExpr::No,
                            ConstantHasGenerics::Yes,
                            None,
                            |this| this.visit_expr(expr),
                        )
                    });
                }
            }
            AssocItemKind::Fn(box Fn { generics, .. }) => {
                debug!("resolve_implementation AssocItemKind::Fn");
                // We also need a new scope for the impl item type parameters.
                self.with_generic_param_rib(
                    &generics.params,
                    AssocItemRibKind,
                    LifetimeRibKind::Generics {
                        binder: item.id,
                        span: generics.span,
                        kind: LifetimeBinderKind::Function,
                    },
                    |this| {
                        // If this is a trait impl, ensure the method
                        // exists in trait
                        this.check_trait_item(
                            item.id,
                            item.ident,
                            &item.kind,
                            ValueNS,
                            item.span,
                            seen_trait_items,
                            |i, s, c| MethodNotMemberOfTrait(i, s, c),
                        );

                        visit::walk_assoc_item(this, item, AssocCtxt::Impl)
                    },
                );
            }
            AssocItemKind::Type(box TyAlias { generics, .. }) => {
                debug!("resolve_implementation AssocItemKind::Type");
                // We also need a new scope for the impl item type parameters.
                self.with_generic_param_rib(
                    &generics.params,
                    AssocItemRibKind,
                    LifetimeRibKind::Generics {
                        binder: item.id,
                        span: generics.span,
                        kind: LifetimeBinderKind::Item,
                    },
                    |this| {
                        this.with_lifetime_rib(LifetimeRibKind::AnonymousReportError, |this| {
                            // If this is a trait impl, ensure the type
                            // exists in trait
                            this.check_trait_item(
                                item.id,
                                item.ident,
                                &item.kind,
                                TypeNS,
                                item.span,
                                seen_trait_items,
                                |i, s, c| TypeNotMemberOfTrait(i, s, c),
                            );

                            visit::walk_assoc_item(this, item, AssocCtxt::Impl)
                        });
                    },
                );
            }
            AssocItemKind::MacCall(_) => {
                panic!("unexpanded macro in resolve!")
            }
        }
    }

    fn check_trait_item<F>(
        &mut self,
        id: NodeId,
        mut ident: Ident,
        kind: &AssocItemKind,
        ns: Namespace,
        span: Span,
        seen_trait_items: &mut FxHashMap<DefId, Span>,
        err: F,
    ) where
        F: FnOnce(Ident, String, Option<Symbol>) -> ResolutionError<'a>,
    {
        // If there is a TraitRef in scope for an impl, then the method must be in the trait.
        let Some((module, _)) = &self.current_trait_ref else { return; };
        ident.span.normalize_to_macros_2_0_and_adjust(module.expansion);
        let key = self.r.new_key(ident, ns);
        let mut binding = self.r.resolution(module, key).try_borrow().ok().and_then(|r| r.binding);
        debug!(?binding);
        if binding.is_none() {
            // We could not find the trait item in the correct namespace.
            // Check the other namespace to report an error.
            let ns = match ns {
                ValueNS => TypeNS,
                TypeNS => ValueNS,
                _ => ns,
            };
            let key = self.r.new_key(ident, ns);
            binding = self.r.resolution(module, key).try_borrow().ok().and_then(|r| r.binding);
            debug!(?binding);
        }
        let Some(binding) = binding else {
            // We could not find the method: report an error.
            let candidate = self.find_similarly_named_assoc_item(ident.name, kind);
            let path = &self.current_trait_ref.as_ref().unwrap().1.path;
            let path_names = path_names_to_string(path);
            self.report_error(span, err(ident, path_names, candidate));
            return;
        };

        let res = binding.res();
        let Res::Def(def_kind, id_in_trait) = res else { bug!() };

        match seen_trait_items.entry(id_in_trait) {
            Entry::Occupied(entry) => {
                self.report_error(
                    span,
                    ResolutionError::TraitImplDuplicate {
                        name: ident.name,
                        old_span: *entry.get(),
                        trait_item_span: binding.span,
                    },
                );
                return;
            }
            Entry::Vacant(entry) => {
                entry.insert(span);
            }
        };

        match (def_kind, kind) {
            (DefKind::AssocTy, AssocItemKind::Type(..))
            | (DefKind::AssocFn, AssocItemKind::Fn(..))
            | (DefKind::AssocConst, AssocItemKind::Const(..)) => {
                self.r.record_partial_res(id, PartialRes::new(res));
                return;
            }
            _ => {}
        }

        // The method kind does not correspond to what appeared in the trait, report.
        let path = &self.current_trait_ref.as_ref().unwrap().1.path;
        let (code, kind) = match kind {
            AssocItemKind::Const(..) => (rustc_errors::error_code!(E0323), "const"),
            AssocItemKind::Fn(..) => (rustc_errors::error_code!(E0324), "method"),
            AssocItemKind::Type(..) => (rustc_errors::error_code!(E0325), "type"),
            AssocItemKind::MacCall(..) => span_bug!(span, "unexpanded macro"),
        };
        let trait_path = path_names_to_string(path);
        self.report_error(
            span,
            ResolutionError::TraitImplMismatch {
                name: ident.name,
                kind,
                code,
                trait_path,
                trait_item_span: binding.span,
            },
        );
    }

    fn resolve_params(&mut self, params: &'ast [Param]) {
        let mut bindings = smallvec![(PatBoundCtx::Product, Default::default())];
        self.with_lifetime_rib(LifetimeRibKind::Elided(LifetimeRes::Infer), |this| {
            for Param { pat, .. } in params {
                this.resolve_pattern(pat, PatternSource::FnParam, &mut bindings);
            }
        });
        for Param { ty, .. } in params {
            self.visit_ty(ty);
        }
    }

    fn resolve_local(&mut self, local: &'ast Local) {
        debug!("resolving local ({:?})", local);
        // Resolve the type.
        walk_list!(self, visit_ty, &local.ty);

        // Resolve the initializer.
        if let Some((init, els)) = local.kind.init_else_opt() {
            self.visit_expr(init);

            // Resolve the `else` block
            if let Some(els) = els {
                self.visit_block(els);
            }
        }

        // Resolve the pattern.
        self.resolve_pattern_top(&local.pat, PatternSource::Let);
    }

    /// build a map from pattern identifiers to binding-info's.
    /// this is done hygienically. This could arise for a macro
    /// that expands into an or-pattern where one 'x' was from the
    /// user and one 'x' came from the macro.
    fn binding_mode_map(&mut self, pat: &Pat) -> BindingMap {
        let mut binding_map = FxHashMap::default();

        pat.walk(&mut |pat| {
            match pat.kind {
                PatKind::Ident(annotation, ident, ref sub_pat)
                    if sub_pat.is_some() || self.is_base_res_local(pat.id) =>
                {
                    binding_map.insert(ident, BindingInfo { span: ident.span, annotation });
                }
                PatKind::Or(ref ps) => {
                    // Check the consistency of this or-pattern and
                    // then add all bindings to the larger map.
                    for bm in self.check_consistent_bindings(ps) {
                        binding_map.extend(bm);
                    }
                    return false;
                }
                _ => {}
            }

            true
        });

        binding_map
    }

    fn is_base_res_local(&self, nid: NodeId) -> bool {
        matches!(
            self.r.partial_res_map.get(&nid).map(|res| res.expect_full_res()),
            Some(Res::Local(..))
        )
    }

    /// Checks that all of the arms in an or-pattern have exactly the
    /// same set of bindings, with the same binding modes for each.
    fn check_consistent_bindings(&mut self, pats: &[P<Pat>]) -> Vec<BindingMap> {
        let mut missing_vars = FxHashMap::default();
        let mut inconsistent_vars = FxHashMap::default();

        // 1) Compute the binding maps of all arms.
        let maps = pats.iter().map(|pat| self.binding_mode_map(pat)).collect::<Vec<_>>();

        // 2) Record any missing bindings or binding mode inconsistencies.
        for (map_outer, pat_outer) in pats.iter().enumerate().map(|(idx, pat)| (&maps[idx], pat)) {
            // Check against all arms except for the same pattern which is always self-consistent.
            let inners = pats
                .iter()
                .enumerate()
                .filter(|(_, pat)| pat.id != pat_outer.id)
                .flat_map(|(idx, _)| maps[idx].iter())
                .map(|(key, binding)| (key.name, map_outer.get(&key), binding));

            for (name, info, &binding_inner) in inners {
                match info {
                    None => {
                        // The inner binding is missing in the outer.
                        let binding_error =
                            missing_vars.entry(name).or_insert_with(|| BindingError {
                                name,
                                origin: BTreeSet::new(),
                                target: BTreeSet::new(),
                                could_be_path: name.as_str().starts_with(char::is_uppercase),
                            });
                        binding_error.origin.insert(binding_inner.span);
                        binding_error.target.insert(pat_outer.span);
                    }
                    Some(binding_outer) => {
                        if binding_outer.annotation != binding_inner.annotation {
                            // The binding modes in the outer and inner bindings differ.
                            inconsistent_vars
                                .entry(name)
                                .or_insert((binding_inner.span, binding_outer.span));
                        }
                    }
                }
            }
        }

        // 3) Report all missing variables we found.
        let mut missing_vars = missing_vars.into_iter().collect::<Vec<_>>();
        missing_vars.sort_by_key(|&(sym, ref _err)| sym);

        for (name, mut v) in missing_vars.into_iter() {
            if inconsistent_vars.contains_key(&name) {
                v.could_be_path = false;
            }
            self.report_error(
                *v.origin.iter().next().unwrap(),
                ResolutionError::VariableNotBoundInPattern(v, self.parent_scope),
            );
        }

        // 4) Report all inconsistencies in binding modes we found.
        let mut inconsistent_vars = inconsistent_vars.iter().collect::<Vec<_>>();
        inconsistent_vars.sort();
        for (name, v) in inconsistent_vars {
            self.report_error(v.0, ResolutionError::VariableBoundWithDifferentMode(*name, v.1));
        }

        // 5) Finally bubble up all the binding maps.
        maps
    }

    /// Check the consistency of the outermost or-patterns.
    fn check_consistent_bindings_top(&mut self, pat: &'ast Pat) {
        pat.walk(&mut |pat| match pat.kind {
            PatKind::Or(ref ps) => {
                self.check_consistent_bindings(ps);
                false
            }
            _ => true,
        })
    }

    fn resolve_arm(&mut self, arm: &'ast Arm) {
        self.with_rib(ValueNS, NormalRibKind, |this| {
            this.resolve_pattern_top(&arm.pat, PatternSource::Match);
            walk_list!(this, visit_expr, &arm.guard);
            this.visit_expr(&arm.body);
        });
    }

    /// Arising from `source`, resolve a top level pattern.
    fn resolve_pattern_top(&mut self, pat: &'ast Pat, pat_src: PatternSource) {
        let mut bindings = smallvec![(PatBoundCtx::Product, Default::default())];
        self.resolve_pattern(pat, pat_src, &mut bindings);
    }

    fn resolve_pattern(
        &mut self,
        pat: &'ast Pat,
        pat_src: PatternSource,
        bindings: &mut SmallVec<[(PatBoundCtx, FxHashSet<Ident>); 1]>,
    ) {
        // We walk the pattern before declaring the pattern's inner bindings,
        // so that we avoid resolving a literal expression to a binding defined
        // by the pattern.
        visit::walk_pat(self, pat);
        self.resolve_pattern_inner(pat, pat_src, bindings);
        // This has to happen *after* we determine which pat_idents are variants:
        self.check_consistent_bindings_top(pat);
    }

    /// Resolve bindings in a pattern. This is a helper to `resolve_pattern`.
    ///
    /// ### `bindings`
    ///
    /// A stack of sets of bindings accumulated.
    ///
    /// In each set, `PatBoundCtx::Product` denotes that a found binding in it should
    /// be interpreted as re-binding an already bound binding. This results in an error.
    /// Meanwhile, `PatBound::Or` denotes that a found binding in the set should result
    /// in reusing this binding rather than creating a fresh one.
    ///
    /// When called at the top level, the stack must have a single element
    /// with `PatBound::Product`. Otherwise, pushing to the stack happens as
    /// or-patterns (`p_0 | ... | p_n`) are encountered and the context needs
    /// to be switched to `PatBoundCtx::Or` and then `PatBoundCtx::Product` for each `p_i`.
    /// When each `p_i` has been dealt with, the top set is merged with its parent.
    /// When a whole or-pattern has been dealt with, the thing happens.
    ///
    /// See the implementation and `fresh_binding` for more details.
    fn resolve_pattern_inner(
        &mut self,
        pat: &Pat,
        pat_src: PatternSource,
        bindings: &mut SmallVec<[(PatBoundCtx, FxHashSet<Ident>); 1]>,
    ) {
        // Visit all direct subpatterns of this pattern.
        pat.walk(&mut |pat| {
            debug!("resolve_pattern pat={:?} node={:?}", pat, pat.kind);
            match pat.kind {
                PatKind::Ident(bmode, ident, ref sub) => {
                    // First try to resolve the identifier as some existing entity,
                    // then fall back to a fresh binding.
                    let has_sub = sub.is_some();
                    let res = self
                        .try_resolve_as_non_binding(pat_src, bmode, ident, has_sub)
                        .unwrap_or_else(|| self.fresh_binding(ident, pat.id, pat_src, bindings));
                    self.r.record_partial_res(pat.id, PartialRes::new(res));
                    self.r.record_pat_span(pat.id, pat.span);
                }
                PatKind::TupleStruct(ref qself, ref path, ref sub_patterns) => {
                    self.smart_resolve_path(
                        pat.id,
                        qself,
                        path,
                        PathSource::TupleStruct(
                            pat.span,
                            self.r.arenas.alloc_pattern_spans(sub_patterns.iter().map(|p| p.span)),
                        ),
                    );
                }
                PatKind::Path(ref qself, ref path) => {
                    self.smart_resolve_path(pat.id, qself, path, PathSource::Pat);
                }
                PatKind::Struct(ref qself, ref path, ..) => {
                    self.smart_resolve_path(pat.id, qself, path, PathSource::Struct);
                }
                PatKind::Or(ref ps) => {
                    // Add a new set of bindings to the stack. `Or` here records that when a
                    // binding already exists in this set, it should not result in an error because
                    // `V1(a) | V2(a)` must be allowed and are checked for consistency later.
                    bindings.push((PatBoundCtx::Or, Default::default()));
                    for p in ps {
                        // Now we need to switch back to a product context so that each
                        // part of the or-pattern internally rejects already bound names.
                        // For example, `V1(a) | V2(a, a)` and `V1(a, a) | V2(a)` are bad.
                        bindings.push((PatBoundCtx::Product, Default::default()));
                        self.resolve_pattern_inner(p, pat_src, bindings);
                        // Move up the non-overlapping bindings to the or-pattern.
                        // Existing bindings just get "merged".
                        let collected = bindings.pop().unwrap().1;
                        bindings.last_mut().unwrap().1.extend(collected);
                    }
                    // This or-pattern itself can itself be part of a product,
                    // e.g. `(V1(a) | V2(a), a)` or `(a, V1(a) | V2(a))`.
                    // Both cases bind `a` again in a product pattern and must be rejected.
                    let collected = bindings.pop().unwrap().1;
                    bindings.last_mut().unwrap().1.extend(collected);

                    // Prevent visiting `ps` as we've already done so above.
                    return false;
                }
                _ => {}
            }
            true
        });
    }

    fn fresh_binding(
        &mut self,
        ident: Ident,
        pat_id: NodeId,
        pat_src: PatternSource,
        bindings: &mut SmallVec<[(PatBoundCtx, FxHashSet<Ident>); 1]>,
    ) -> Res {
        // Add the binding to the local ribs, if it doesn't already exist in the bindings map.
        // (We must not add it if it's in the bindings map because that breaks the assumptions
        // later passes make about or-patterns.)
        let ident = ident.normalize_to_macro_rules();

        let mut bound_iter = bindings.iter().filter(|(_, set)| set.contains(&ident));
        // Already bound in a product pattern? e.g. `(a, a)` which is not allowed.
        let already_bound_and = bound_iter.clone().any(|(ctx, _)| *ctx == PatBoundCtx::Product);
        // Already bound in an or-pattern? e.g. `V1(a) | V2(a)`.
        // This is *required* for consistency which is checked later.
        let already_bound_or = bound_iter.any(|(ctx, _)| *ctx == PatBoundCtx::Or);

        if already_bound_and {
            // Overlap in a product pattern somewhere; report an error.
            use ResolutionError::*;
            let error = match pat_src {
                // `fn f(a: u8, a: u8)`:
                PatternSource::FnParam => IdentifierBoundMoreThanOnceInParameterList,
                // `Variant(a, a)`:
                _ => IdentifierBoundMoreThanOnceInSamePattern,
            };
            self.report_error(ident.span, error(ident.name));
        }

        // Record as bound if it's valid:
        let ident_valid = ident.name != kw::Empty;
        if ident_valid {
            bindings.last_mut().unwrap().1.insert(ident);
        }

        if already_bound_or {
            // `Variant1(a) | Variant2(a)`, ok
            // Reuse definition from the first `a`.
            self.innermost_rib_bindings(ValueNS)[&ident]
        } else {
            let res = Res::Local(pat_id);
            if ident_valid {
                // A completely fresh binding add to the set if it's valid.
                self.innermost_rib_bindings(ValueNS).insert(ident, res);
            }
            res
        }
    }

    fn innermost_rib_bindings(&mut self, ns: Namespace) -> &mut IdentMap<Res> {
        &mut self.ribs[ns].last_mut().unwrap().bindings
    }

    fn try_resolve_as_non_binding(
        &mut self,
        pat_src: PatternSource,
        ann: BindingAnnotation,
        ident: Ident,
        has_sub: bool,
    ) -> Option<Res> {
        // An immutable (no `mut`) by-value (no `ref`) binding pattern without
        // a sub pattern (no `@ $pat`) is syntactically ambiguous as it could
        // also be interpreted as a path to e.g. a constant, variant, etc.
        let is_syntactic_ambiguity = !has_sub && ann == BindingAnnotation::NONE;

        let ls_binding = self.maybe_resolve_ident_in_lexical_scope(ident, ValueNS)?;
        let (res, binding) = match ls_binding {
            LexicalScopeBinding::Item(binding)
                if is_syntactic_ambiguity && binding.is_ambiguity() =>
            {
                // For ambiguous bindings we don't know all their definitions and cannot check
                // whether they can be shadowed by fresh bindings or not, so force an error.
                // issues/33118#issuecomment-233962221 (see below) still applies here,
                // but we have to ignore it for backward compatibility.
                self.r.record_use(ident, binding, false);
                return None;
            }
            LexicalScopeBinding::Item(binding) => (binding.res(), Some(binding)),
            LexicalScopeBinding::Res(res) => (res, None),
        };

        match res {
            Res::SelfCtor(_) // See #70549.
            | Res::Def(
                DefKind::Ctor(_, CtorKind::Const) | DefKind::Const | DefKind::ConstParam,
                _,
            ) if is_syntactic_ambiguity => {
                // Disambiguate in favor of a unit struct/variant or constant pattern.
                if let Some(binding) = binding {
                    self.r.record_use(ident, binding, false);
                }
                Some(res)
            }
            Res::Def(DefKind::Ctor(..) | DefKind::Const | DefKind::Static(_), _) => {
                // This is unambiguously a fresh binding, either syntactically
                // (e.g., `IDENT @ PAT` or `ref IDENT`) or because `IDENT` resolves
                // to something unusable as a pattern (e.g., constructor function),
                // but we still conservatively report an error, see
                // issues/33118#issuecomment-233962221 for one reason why.
                let binding = binding.expect("no binding for a ctor or static");
                self.report_error(
                    ident.span,
                    ResolutionError::BindingShadowsSomethingUnacceptable {
                        shadowing_binding: pat_src,
                        name: ident.name,
                        participle: if binding.is_import() { "imported" } else { "defined" },
                        article: binding.res().article(),
                        shadowed_binding: binding.res(),
                        shadowed_binding_span: binding.span,
                    },
                );
                None
            }
            Res::Def(DefKind::ConstParam, def_id) => {
                // Same as for DefKind::Const above, but here, `binding` is `None`, so we
                // have to construct the error differently
                self.report_error(
                    ident.span,
                    ResolutionError::BindingShadowsSomethingUnacceptable {
                        shadowing_binding: pat_src,
                        name: ident.name,
                        participle: "defined",
                        article: res.article(),
                        shadowed_binding: res,
                        shadowed_binding_span: self.r.def_span(def_id),
                    }
                );
                None
            }
            Res::Def(DefKind::Fn, _) | Res::Local(..) | Res::Err => {
                // These entities are explicitly allowed to be shadowed by fresh bindings.
                None
            }
            Res::SelfCtor(_) => {
                // We resolve `Self` in pattern position as an ident sometimes during recovery,
                // so delay a bug instead of ICEing.
                self.r.tcx.sess.delay_span_bug(
                    ident.span,
                    "unexpected `SelfCtor` in pattern, expected identifier"
                );
                None
            }
            _ => span_bug!(
                ident.span,
                "unexpected resolution for an identifier in pattern: {:?}",
                res,
            ),
        }
    }

    // High-level and context dependent path resolution routine.
    // Resolves the path and records the resolution into definition map.
    // If resolution fails tries several techniques to find likely
    // resolution candidates, suggest imports or other help, and report
    // errors in user friendly way.
    fn smart_resolve_path(
        &mut self,
        id: NodeId,
        qself: &Option<P<QSelf>>,
        path: &Path,
        source: PathSource<'ast>,
    ) {
        self.smart_resolve_path_fragment(
            qself,
            &Segment::from_path(path),
            source,
            Finalize::new(id, path.span),
        );
    }

    #[instrument(level = "debug", skip(self))]
    fn smart_resolve_path_fragment(
        &mut self,
        qself: &Option<P<QSelf>>,
        path: &[Segment],
        source: PathSource<'ast>,
        finalize: Finalize,
    ) -> PartialRes {
        let ns = source.namespace();

        let Finalize { node_id, path_span, .. } = finalize;
        let report_errors = |this: &mut Self, res: Option<Res>| {
            if this.should_report_errs() {
                let (err, candidates) =
                    this.smart_resolve_report_errors(path, path_span, source, res);

                let def_id = this.parent_scope.module.nearest_parent_mod();
                let instead = res.is_some();
                let suggestion = if let Some((start, end)) = this.diagnostic_metadata.in_range
                    && path[0].ident.span.lo() == end.span.lo()
                {
                    let mut sugg = ".";
                    let mut span = start.span.between(end.span);
                    if span.lo() + BytePos(2) == span.hi() {
                        // There's no space between the start, the range op and the end, suggest
                        // removal which will look better.
                        span = span.with_lo(span.lo() + BytePos(1));
                        sugg = "";
                    }
                    Some((
                        span,
                        "you might have meant to write `.` instead of `..`",
                        sugg.to_string(),
                        Applicability::MaybeIncorrect,
                    ))
                } else if res.is_none() && matches!(source, PathSource::Type) {
                    this.report_missing_type_error(path)
                } else {
                    None
                };

                this.r.use_injections.push(UseError {
                    err,
                    candidates,
                    def_id,
                    instead,
                    suggestion,
                    path: path.into(),
                    is_call: source.is_call(),
                });
            }

            PartialRes::new(Res::Err)
        };

        // For paths originating from calls (like in `HashMap::new()`), tries
        // to enrich the plain `failed to resolve: ...` message with hints
        // about possible missing imports.
        //
        // Similar thing, for types, happens in `report_errors` above.
        let report_errors_for_call = |this: &mut Self, parent_err: Spanned<ResolutionError<'a>>| {
            if !source.is_call() {
                return Some(parent_err);
            }

            // Before we start looking for candidates, we have to get our hands
            // on the type user is trying to perform invocation on; basically:
            // we're transforming `HashMap::new` into just `HashMap`.
            let prefix_path = match path.split_last() {
                Some((_, path)) if !path.is_empty() => path,
                _ => return Some(parent_err),
            };

            let (mut err, candidates) =
                this.smart_resolve_report_errors(prefix_path, path_span, PathSource::Type, None);

            // There are two different error messages user might receive at
            // this point:
            // - E0412 cannot find type `{}` in this scope
            // - E0433 failed to resolve: use of undeclared type or module `{}`
            //
            // The first one is emitted for paths in type-position, and the
            // latter one - for paths in expression-position.
            //
            // Thus (since we're in expression-position at this point), not to
            // confuse the user, we want to keep the *message* from E0433 (so
            // `parent_err`), but we want *hints* from E0412 (so `err`).
            //
            // And that's what happens below - we're just mixing both messages
            // into a single one.
            let mut parent_err = this.r.into_struct_error(parent_err.span, parent_err.node);

            // overwrite all properties with the parent's error message
            err.message = take(&mut parent_err.message);
            err.code = take(&mut parent_err.code);
            swap(&mut err.span, &mut parent_err.span);
            err.children = take(&mut parent_err.children);
            err.sort_span = parent_err.sort_span;
            err.is_lint = parent_err.is_lint;

            // merge the parent's suggestions with the typo suggestions
            fn append_result<T, E>(res1: &mut Result<Vec<T>, E>, res2: Result<Vec<T>, E>) {
                match res1 {
                    Ok(vec1) => match res2 {
                        Ok(mut vec2) => vec1.append(&mut vec2),
                        Err(e) => *res1 = Err(e),
                    },
                    Err(_) => (),
                };
            }
            append_result(&mut err.suggestions, parent_err.suggestions.clone());

            parent_err.cancel();

            let def_id = this.parent_scope.module.nearest_parent_mod();

            if this.should_report_errs() {
                if candidates.is_empty() {
                    if path.len() == 2 && prefix_path.len() == 1 {
                        // Delay to check whether methond name is an associated function or not
                        // ```
                        // let foo = Foo {};
                        // foo::bar(); // possibly suggest to foo.bar();
                        //```
                        err.stash(
                            prefix_path[0].ident.span,
                            rustc_errors::StashKey::CallAssocMethod,
                        );
                    } else {
                        // When there is no suggested imports, we can just emit the error
                        // and suggestions immediately. Note that we bypass the usually error
                        // reporting routine (ie via `self.r.report_error`) because we need
                        // to post-process the `ResolutionError` above.
                        err.emit();
                    }
                } else {
                    // If there are suggested imports, the error reporting is delayed
                    this.r.use_injections.push(UseError {
                        err,
                        candidates,
                        def_id,
                        instead: false,
                        suggestion: None,
                        path: prefix_path.into(),
                        is_call: source.is_call(),
                    });
                }
            } else {
                err.cancel();
            }

            // We don't return `Some(parent_err)` here, because the error will
            // be already printed either immediately or as part of the `use` injections
            None
        };

        let partial_res = match self.resolve_qpath_anywhere(
            qself,
            path,
            ns,
            path_span,
            source.defer_to_typeck(),
            finalize,
        ) {
            Ok(Some(partial_res)) if let Some(res) = partial_res.full_res() => {
                if source.is_expected(res) || res == Res::Err {
                    partial_res
                } else {
                    report_errors(self, Some(res))
                }
            }

            Ok(Some(partial_res)) if source.defer_to_typeck() => {
                // Not fully resolved associated item `T::A::B` or `<T as Tr>::A::B`
                // or `<T>::A::B`. If `B` should be resolved in value namespace then
                // it needs to be added to the trait map.
                if ns == ValueNS {
                    let item_name = path.last().unwrap().ident;
                    let traits = self.traits_in_scope(item_name, ns);
                    self.r.trait_map.insert(node_id, traits);
                }

                if PrimTy::from_name(path[0].ident.name).is_some() {
                    let mut std_path = Vec::with_capacity(1 + path.len());

                    std_path.push(Segment::from_ident(Ident::with_dummy_span(sym::std)));
                    std_path.extend(path);
                    if let PathResult::Module(_) | PathResult::NonModule(_) =
                        self.resolve_path(&std_path, Some(ns), None)
                    {
                        // Check if we wrote `str::from_utf8` instead of `std::str::from_utf8`
                        let item_span =
                            path.iter().last().map_or(path_span, |segment| segment.ident.span);

                        self.r.confused_type_with_std_module.insert(item_span, path_span);
                        self.r.confused_type_with_std_module.insert(path_span, path_span);
                    }
                }

                partial_res
            }

            Err(err) => {
                if let Some(err) = report_errors_for_call(self, err) {
                    self.report_error(err.span, err.node);
                }

                PartialRes::new(Res::Err)
            }

            _ => report_errors(self, None),
        };

        if !matches!(source, PathSource::TraitItem(..)) {
            // Avoid recording definition of `A::B` in `<T as A>::B::C`.
            self.r.record_partial_res(node_id, partial_res);
            self.resolve_elided_lifetimes_in_path(node_id, partial_res, path, source, path_span);
        }

        partial_res
    }

    fn self_type_is_available(&mut self) -> bool {
        let binding = self
            .maybe_resolve_ident_in_lexical_scope(Ident::with_dummy_span(kw::SelfUpper), TypeNS);
        if let Some(LexicalScopeBinding::Res(res)) = binding { res != Res::Err } else { false }
    }

    fn self_value_is_available(&mut self, self_span: Span) -> bool {
        let ident = Ident::new(kw::SelfLower, self_span);
        let binding = self.maybe_resolve_ident_in_lexical_scope(ident, ValueNS);
        if let Some(LexicalScopeBinding::Res(res)) = binding { res != Res::Err } else { false }
    }

    /// A wrapper around [`Resolver::report_error`].
    ///
    /// This doesn't emit errors for function bodies if this is rustdoc.
    fn report_error(&mut self, span: Span, resolution_error: ResolutionError<'a>) {
        if self.should_report_errs() {
            self.r.report_error(span, resolution_error);
        }
    }

    #[inline]
    /// If we're actually rustdoc then avoid giving a name resolution error for `cfg()` items.
    fn should_report_errs(&self) -> bool {
        !(self.r.tcx.sess.opts.actually_rustdoc && self.in_func_body)
    }

    // Resolve in alternative namespaces if resolution in the primary namespace fails.
    fn resolve_qpath_anywhere(
        &mut self,
        qself: &Option<P<QSelf>>,
        path: &[Segment],
        primary_ns: Namespace,
        span: Span,
        defer_to_typeck: bool,
        finalize: Finalize,
    ) -> Result<Option<PartialRes>, Spanned<ResolutionError<'a>>> {
        let mut fin_res = None;

        for (i, &ns) in [primary_ns, TypeNS, ValueNS].iter().enumerate() {
            if i == 0 || ns != primary_ns {
                match self.resolve_qpath(qself, path, ns, finalize)? {
                    Some(partial_res)
                        if partial_res.unresolved_segments() == 0 || defer_to_typeck =>
                    {
                        return Ok(Some(partial_res));
                    }
                    partial_res => {
                        if fin_res.is_none() {
                            fin_res = partial_res;
                        }
                    }
                }
            }
        }

        assert!(primary_ns != MacroNS);

        if qself.is_none() {
            let path_seg = |seg: &Segment| PathSegment::from_ident(seg.ident);
            let path = Path { segments: path.iter().map(path_seg).collect(), span, tokens: None };
            if let Ok((_, res)) =
                self.r.resolve_macro_path(&path, None, &self.parent_scope, false, false)
            {
                return Ok(Some(PartialRes::new(res)));
            }
        }

        Ok(fin_res)
    }

    /// Handles paths that may refer to associated items.
    fn resolve_qpath(
        &mut self,
        qself: &Option<P<QSelf>>,
        path: &[Segment],
        ns: Namespace,
        finalize: Finalize,
    ) -> Result<Option<PartialRes>, Spanned<ResolutionError<'a>>> {
        debug!(
            "resolve_qpath(qself={:?}, path={:?}, ns={:?}, finalize={:?})",
            qself, path, ns, finalize,
        );

        if let Some(qself) = qself {
            if qself.position == 0 {
                // This is a case like `<T>::B`, where there is no
                // trait to resolve. In that case, we leave the `B`
                // segment to be resolved by type-check.
                return Ok(Some(PartialRes::with_unresolved_segments(
                    Res::Def(DefKind::Mod, CRATE_DEF_ID.to_def_id()),
                    path.len(),
                )));
            }

            // Make sure `A::B` in `<T as A::B>::C` is a trait item.
            //
            // Currently, `path` names the full item (`A::B::C`, in
            // our example). so we extract the prefix of that that is
            // the trait (the slice upto and including
            // `qself.position`). And then we recursively resolve that,
            // but with `qself` set to `None`.
            let ns = if qself.position + 1 == path.len() { ns } else { TypeNS };
            let partial_res = self.smart_resolve_path_fragment(
                &None,
                &path[..=qself.position],
                PathSource::TraitItem(ns),
                Finalize::with_root_span(finalize.node_id, finalize.path_span, qself.path_span),
            );

            // The remaining segments (the `C` in our example) will
            // have to be resolved by type-check, since that requires doing
            // trait resolution.
            return Ok(Some(PartialRes::with_unresolved_segments(
                partial_res.base_res(),
                partial_res.unresolved_segments() + path.len() - qself.position - 1,
            )));
        }

        let result = match self.resolve_path(&path, Some(ns), Some(finalize)) {
            PathResult::NonModule(path_res) => path_res,
            PathResult::Module(ModuleOrUniformRoot::Module(module)) if !module.is_normal() => {
                PartialRes::new(module.res().unwrap())
            }
            // In `a(::assoc_item)*` `a` cannot be a module. If `a` does resolve to a module we
            // don't report an error right away, but try to fallback to a primitive type.
            // So, we are still able to successfully resolve something like
            //
            // use std::u8; // bring module u8 in scope
            // fn f() -> u8 { // OK, resolves to primitive u8, not to std::u8
            //     u8::max_value() // OK, resolves to associated function <u8>::max_value,
            //                     // not to non-existent std::u8::max_value
            // }
            //
            // Such behavior is required for backward compatibility.
            // The same fallback is used when `a` resolves to nothing.
            PathResult::Module(ModuleOrUniformRoot::Module(_)) | PathResult::Failed { .. }
                if (ns == TypeNS || path.len() > 1)
                    && PrimTy::from_name(path[0].ident.name).is_some() =>
            {
                let prim = PrimTy::from_name(path[0].ident.name).unwrap();
                PartialRes::with_unresolved_segments(Res::PrimTy(prim), path.len() - 1)
            }
            PathResult::Module(ModuleOrUniformRoot::Module(module)) => {
                PartialRes::new(module.res().unwrap())
            }
            PathResult::Failed { is_error_from_last_segment: false, span, label, suggestion } => {
                return Err(respan(span, ResolutionError::FailedToResolve { label, suggestion }));
            }
            PathResult::Module(..) | PathResult::Failed { .. } => return Ok(None),
            PathResult::Indeterminate => bug!("indeterminate path result in resolve_qpath"),
        };

        if path.len() > 1
            && let Some(res) = result.full_res()
            && res != Res::Err
            && path[0].ident.name != kw::PathRoot
            && path[0].ident.name != kw::DollarCrate
        {
            let unqualified_result = {
                match self.resolve_path(&[*path.last().unwrap()], Some(ns), None) {
                    PathResult::NonModule(path_res) => path_res.expect_full_res(),
                    PathResult::Module(ModuleOrUniformRoot::Module(module)) => {
                        module.res().unwrap()
                    }
                    _ => return Ok(Some(result)),
                }
            };
            if res == unqualified_result {
                let lint = lint::builtin::UNUSED_QUALIFICATIONS;
                self.r.lint_buffer.buffer_lint(
                    lint,
                    finalize.node_id,
                    finalize.path_span,
                    "unnecessary qualification",
                )
            }
        }

        Ok(Some(result))
    }

    fn with_resolved_label(&mut self, label: Option<Label>, id: NodeId, f: impl FnOnce(&mut Self)) {
        if let Some(label) = label {
            if label.ident.as_str().as_bytes()[1] != b'_' {
                self.diagnostic_metadata.unused_labels.insert(id, label.ident.span);
            }

            if let Ok((_, orig_span)) = self.resolve_label(label.ident) {
                diagnostics::signal_label_shadowing(self.r.tcx.sess, orig_span, label.ident)
            }

            self.with_label_rib(NormalRibKind, |this| {
                let ident = label.ident.normalize_to_macro_rules();
                this.label_ribs.last_mut().unwrap().bindings.insert(ident, id);
                f(this);
            });
        } else {
            f(self);
        }
    }

    fn resolve_labeled_block(&mut self, label: Option<Label>, id: NodeId, block: &'ast Block) {
        self.with_resolved_label(label, id, |this| this.visit_block(block));
    }

    fn resolve_block(&mut self, block: &'ast Block) {
        debug!("(resolving block) entering block");
        // Move down in the graph, if there's an anonymous module rooted here.
        let orig_module = self.parent_scope.module;
        let anonymous_module = self.r.block_map.get(&block.id).cloned(); // clones a reference

        let mut num_macro_definition_ribs = 0;
        if let Some(anonymous_module) = anonymous_module {
            debug!("(resolving block) found anonymous module, moving down");
            self.ribs[ValueNS].push(Rib::new(ModuleRibKind(anonymous_module)));
            self.ribs[TypeNS].push(Rib::new(ModuleRibKind(anonymous_module)));
            self.parent_scope.module = anonymous_module;
        } else {
            self.ribs[ValueNS].push(Rib::new(NormalRibKind));
        }

        let prev = self.diagnostic_metadata.current_block_could_be_bare_struct_literal.take();
        if let (true, [Stmt { kind: StmtKind::Expr(expr), .. }]) =
            (block.could_be_bare_literal, &block.stmts[..])
            && let ExprKind::Type(..) = expr.kind
        {
            self.diagnostic_metadata.current_block_could_be_bare_struct_literal =
            Some(block.span);
        }
        // Descend into the block.
        for stmt in &block.stmts {
            if let StmtKind::Item(ref item) = stmt.kind
                && let ItemKind::MacroDef(..) = item.kind {
                num_macro_definition_ribs += 1;
                let res = self.r.local_def_id(item.id).to_def_id();
                self.ribs[ValueNS].push(Rib::new(MacroDefinition(res)));
                self.label_ribs.push(Rib::new(MacroDefinition(res)));
            }

            self.visit_stmt(stmt);
        }
        self.diagnostic_metadata.current_block_could_be_bare_struct_literal = prev;

        // Move back up.
        self.parent_scope.module = orig_module;
        for _ in 0..num_macro_definition_ribs {
            self.ribs[ValueNS].pop();
            self.label_ribs.pop();
        }
        self.last_block_rib = self.ribs[ValueNS].pop();
        if anonymous_module.is_some() {
            self.ribs[TypeNS].pop();
        }
        debug!("(resolving block) leaving block");
    }

    fn resolve_anon_const(&mut self, constant: &'ast AnonConst, is_repeat: IsRepeatExpr) {
        debug!("resolve_anon_const {:?} is_repeat: {:?}", constant, is_repeat);
        self.with_constant_rib(
            is_repeat,
            if constant.value.is_potential_trivial_const_param() {
                ConstantHasGenerics::Yes
            } else {
                ConstantHasGenerics::No
            },
            None,
            |this| visit::walk_anon_const(this, constant),
        );
    }

    fn resolve_inline_const(&mut self, constant: &'ast AnonConst) {
        debug!("resolve_anon_const {constant:?}");
        self.with_constant_rib(IsRepeatExpr::No, ConstantHasGenerics::Yes, None, |this| {
            visit::walk_anon_const(this, constant)
        });
    }

    fn resolve_expr(&mut self, expr: &'ast Expr, parent: Option<&'ast Expr>) {
        // First, record candidate traits for this expression if it could
        // result in the invocation of a method call.

        self.record_candidate_traits_for_expr_if_necessary(expr);

        // Next, resolve the node.
        match expr.kind {
            ExprKind::Path(ref qself, ref path) => {
                self.smart_resolve_path(expr.id, qself, path, PathSource::Expr(parent));
                visit::walk_expr(self, expr);
            }

            ExprKind::Struct(ref se) => {
                self.smart_resolve_path(expr.id, &se.qself, &se.path, PathSource::Struct);
                visit::walk_expr(self, expr);
            }

            ExprKind::Break(Some(label), _) | ExprKind::Continue(Some(label)) => {
                match self.resolve_label(label.ident) {
                    Ok((node_id, _)) => {
                        // Since this res is a label, it is never read.
                        self.r.label_res_map.insert(expr.id, node_id);
                        self.diagnostic_metadata.unused_labels.remove(&node_id);
                    }
                    Err(error) => {
                        self.report_error(label.ident.span, error);
                    }
                }

                // visit `break` argument if any
                visit::walk_expr(self, expr);
            }

            ExprKind::Break(None, Some(ref e)) => {
                // We use this instead of `visit::walk_expr` to keep the parent expr around for
                // better diagnostics.
                self.resolve_expr(e, Some(&expr));
            }

            ExprKind::Let(ref pat, ref scrutinee, _) => {
                self.visit_expr(scrutinee);
                self.resolve_pattern_top(pat, PatternSource::Let);
            }

            ExprKind::If(ref cond, ref then, ref opt_else) => {
                self.with_rib(ValueNS, NormalRibKind, |this| {
                    let old = this.diagnostic_metadata.in_if_condition.replace(cond);
                    this.visit_expr(cond);
                    this.diagnostic_metadata.in_if_condition = old;
                    this.visit_block(then);
                });
                if let Some(expr) = opt_else {
                    self.visit_expr(expr);
                }
            }

            ExprKind::Loop(ref block, label, _) => {
                self.resolve_labeled_block(label, expr.id, &block)
            }

            ExprKind::While(ref cond, ref block, label) => {
                self.with_resolved_label(label, expr.id, |this| {
                    this.with_rib(ValueNS, NormalRibKind, |this| {
                        let old = this.diagnostic_metadata.in_if_condition.replace(cond);
                        this.visit_expr(cond);
                        this.diagnostic_metadata.in_if_condition = old;
                        this.visit_block(block);
                    })
                });
            }

            ExprKind::ForLoop(ref pat, ref iter_expr, ref block, label) => {
                self.visit_expr(iter_expr);
                self.with_rib(ValueNS, NormalRibKind, |this| {
                    this.resolve_pattern_top(pat, PatternSource::For);
                    this.resolve_labeled_block(label, expr.id, block);
                });
            }

            ExprKind::Block(ref block, label) => self.resolve_labeled_block(label, block.id, block),

            // Equivalent to `visit::walk_expr` + passing some context to children.
            ExprKind::Field(ref subexpression, _) => {
                self.resolve_expr(subexpression, Some(expr));
            }
            ExprKind::MethodCall(box MethodCall { ref seg, ref receiver, ref args, .. }) => {
                self.resolve_expr(receiver, Some(expr));
                for arg in args {
                    self.resolve_expr(arg, None);
                }
                self.visit_path_segment(seg);
            }

            ExprKind::Call(ref callee, ref arguments) => {
                self.resolve_expr(callee, Some(expr));
                let const_args = self.r.legacy_const_generic_args(callee).unwrap_or_default();
                for (idx, argument) in arguments.iter().enumerate() {
                    // Constant arguments need to be treated as AnonConst since
                    // that is how they will be later lowered to HIR.
                    if const_args.contains(&idx) {
                        self.with_constant_rib(
                            IsRepeatExpr::No,
                            if argument.is_potential_trivial_const_param() {
                                ConstantHasGenerics::Yes
                            } else {
                                ConstantHasGenerics::No
                            },
                            None,
                            |this| {
                                this.resolve_expr(argument, None);
                            },
                        );
                    } else {
                        self.resolve_expr(argument, None);
                    }
                }
            }
            ExprKind::Type(ref type_expr, ref ty) => {
                // `ParseSess::type_ascription_path_suggestions` keeps spans of colon tokens in
                // type ascription. Here we are trying to retrieve the span of the colon token as
                // well, but only if it's written without spaces `expr:Ty` and therefore confusable
                // with `expr::Ty`, only in this case it will match the span from
                // `type_ascription_path_suggestions`.
                self.diagnostic_metadata
                    .current_type_ascription
                    .push(type_expr.span.between(ty.span));
                visit::walk_expr(self, expr);
                self.diagnostic_metadata.current_type_ascription.pop();
            }
            // `async |x| ...` gets desugared to `|x| async {...}`, so we need to
            // resolve the arguments within the proper scopes so that usages of them inside the
            // closure are detected as upvars rather than normal closure arg usages.
            ExprKind::Closure(box ast::Closure {
                asyncness: Async::Yes { .. },
                ref fn_decl,
                ref body,
                ..
            }) => {
                self.with_rib(ValueNS, NormalRibKind, |this| {
                    this.with_label_rib(ClosureOrAsyncRibKind, |this| {
                        // Resolve arguments:
                        this.resolve_params(&fn_decl.inputs);
                        // No need to resolve return type --
                        // the outer closure return type is `FnRetTy::Default`.

                        // Now resolve the inner closure
                        {
                            // No need to resolve arguments: the inner closure has none.
                            // Resolve the return type:
                            visit::walk_fn_ret_ty(this, &fn_decl.output);
                            // Resolve the body
                            this.visit_expr(body);
                        }
                    })
                });
            }
            // For closures, ClosureOrAsyncRibKind is added in visit_fn
            ExprKind::Closure(box ast::Closure {
                binder: ClosureBinder::For { ref generic_params, span },
                ..
            }) => {
                self.with_generic_param_rib(
                    &generic_params,
                    NormalRibKind,
                    LifetimeRibKind::Generics {
                        binder: expr.id,
                        kind: LifetimeBinderKind::Closure,
                        span,
                    },
                    |this| visit::walk_expr(this, expr),
                );
            }
            ExprKind::Closure(..) => visit::walk_expr(self, expr),
            ExprKind::Async(..) => {
                self.with_label_rib(ClosureOrAsyncRibKind, |this| visit::walk_expr(this, expr));
            }
            ExprKind::Repeat(ref elem, ref ct) => {
                self.visit_expr(elem);
                self.with_lifetime_rib(LifetimeRibKind::AnonConst, |this| {
                    this.with_lifetime_rib(LifetimeRibKind::Elided(LifetimeRes::Static), |this| {
                        this.resolve_anon_const(ct, IsRepeatExpr::Yes)
                    })
                });
            }
            ExprKind::ConstBlock(ref ct) => {
                self.resolve_inline_const(ct);
            }
            ExprKind::Index(ref elem, ref idx) => {
                self.resolve_expr(elem, Some(expr));
                self.visit_expr(idx);
            }
            ExprKind::Assign(ref lhs, ref rhs, _) => {
                if !self.diagnostic_metadata.is_assign_rhs {
                    self.diagnostic_metadata.in_assignment = Some(expr);
                }
                self.visit_expr(lhs);
                self.diagnostic_metadata.is_assign_rhs = true;
                self.diagnostic_metadata.in_assignment = None;
                self.visit_expr(rhs);
                self.diagnostic_metadata.is_assign_rhs = false;
            }
            ExprKind::Range(Some(ref start), Some(ref end), RangeLimits::HalfOpen) => {
                self.diagnostic_metadata.in_range = Some((start, end));
                self.resolve_expr(start, Some(expr));
                self.resolve_expr(end, Some(expr));
                self.diagnostic_metadata.in_range = None;
            }
            _ => {
                visit::walk_expr(self, expr);
            }
        }
    }

    fn record_candidate_traits_for_expr_if_necessary(&mut self, expr: &'ast Expr) {
        match expr.kind {
            ExprKind::Field(_, ident) => {
                // #6890: Even though you can't treat a method like a field,
                // we need to add any trait methods we find that match the
                // field name so that we can do some nice error reporting
                // later on in typeck.
                let traits = self.traits_in_scope(ident, ValueNS);
                self.r.trait_map.insert(expr.id, traits);
            }
            ExprKind::MethodCall(ref call) => {
                debug!("(recording candidate traits for expr) recording traits for {}", expr.id);
                let traits = self.traits_in_scope(call.seg.ident, ValueNS);
                self.r.trait_map.insert(expr.id, traits);
            }
            _ => {
                // Nothing to do.
            }
        }
    }

    fn traits_in_scope(&mut self, ident: Ident, ns: Namespace) -> Vec<TraitCandidate> {
        self.r.traits_in_scope(
            self.current_trait_ref.as_ref().map(|(module, _)| *module),
            &self.parent_scope,
            ident.span.ctxt(),
            Some((ident.name, ns)),
        )
    }

    /// Construct the list of in-scope lifetime parameters for async lowering.
    /// We include all lifetime parameters, either named or "Fresh".
    /// The order of those parameters does not matter, as long as it is
    /// deterministic.
    fn record_lifetime_params_for_async(
        &mut self,
        fn_id: NodeId,
        async_node_id: Option<(NodeId, Span)>,
    ) {
        if let Some((async_node_id, span)) = async_node_id {
            let mut extra_lifetime_params =
                self.r.extra_lifetime_params_map.get(&fn_id).cloned().unwrap_or_default();
            for rib in self.lifetime_ribs.iter().rev() {
                extra_lifetime_params.extend(
                    rib.bindings.iter().map(|(&ident, &(node_id, res))| (ident, node_id, res)),
                );
                match rib.kind {
                    LifetimeRibKind::Item => break,
                    LifetimeRibKind::AnonymousCreateParameter { binder, .. } => {
                        if let Some(earlier_fresh) = self.r.extra_lifetime_params_map.get(&binder) {
                            extra_lifetime_params.extend(earlier_fresh);
                        }
                    }
                    LifetimeRibKind::Generics { .. } => {}
                    _ => {
                        // We are in a function definition. We should only find `Generics`
                        // and `AnonymousCreateParameter` inside the innermost `Item`.
                        span_bug!(span, "unexpected rib kind: {:?}", rib.kind)
                    }
                }
            }
            self.r.extra_lifetime_params_map.insert(async_node_id, extra_lifetime_params);
        }
    }

    fn resolve_and_cache_rustdoc_path(&mut self, path_str: &str, ns: Namespace) -> bool {
        // FIXME: This caching may be incorrect in case of multiple `macro_rules`
        // items with the same name in the same module.
        // Also hygiene is not considered.
        let mut doc_link_resolutions = std::mem::take(&mut self.r.doc_link_resolutions);
        let res = doc_link_resolutions
            .entry(self.parent_scope.module.nearest_parent_mod().expect_local())
            .or_default()
            .entry((Symbol::intern(path_str), ns))
            .or_insert_with_key(|(path, ns)| {
                let res = self.r.resolve_rustdoc_path(path.as_str(), *ns, self.parent_scope);
                if let Some(res) = res
                    && let Some(def_id) = res.opt_def_id()
                    && !def_id.is_local()
                    && self.r.tcx.sess.crate_types().contains(&CrateType::ProcMacro)
                    && matches!(self.r.tcx.sess.opts.resolve_doc_links, ResolveDocLinks::ExportedMetadata) {
                    // Encoding foreign def ids in proc macro crate metadata will ICE.
                    return None;
                }
                res
            })
            .is_some();
        self.r.doc_link_resolutions = doc_link_resolutions;
        res
    }

    fn resolve_doc_links(&mut self, attrs: &[Attribute], maybe_exported: MaybeExported<'_>) {
        match self.r.tcx.sess.opts.resolve_doc_links {
            ResolveDocLinks::None => return,
            ResolveDocLinks::ExportedMetadata
                if !self.r.tcx.sess.crate_types().iter().copied().any(CrateType::has_metadata)
                    || !maybe_exported.eval(self.r) =>
            {
                return;
            }
            ResolveDocLinks::Exported if !maybe_exported.eval(self.r) => {
                return;
            }
            ResolveDocLinks::ExportedMetadata
            | ResolveDocLinks::Exported
            | ResolveDocLinks::All => {}
        }

        if !attrs.iter().any(|attr| attr.may_have_doc_links()) {
            return;
        }

        let mut need_traits_in_scope = false;
        for path_str in rustdoc::attrs_to_preprocessed_links(attrs) {
            // Resolve all namespaces due to no disambiguator or for diagnostics.
            let mut any_resolved = false;
            let mut need_assoc = false;
            for ns in [TypeNS, ValueNS, MacroNS] {
                if self.resolve_and_cache_rustdoc_path(&path_str, ns) {
                    any_resolved = true;
                } else if ns != MacroNS {
                    need_assoc = true;
                }
            }

            // Resolve all prefixes for type-relative resolution or for diagnostics.
            if need_assoc || !any_resolved {
                let mut path = &path_str[..];
                while let Some(idx) = path.rfind("::") {
                    path = &path[..idx];
                    need_traits_in_scope = true;
                    for ns in [TypeNS, ValueNS, MacroNS] {
                        self.resolve_and_cache_rustdoc_path(path, ns);
                    }
                }
            }
        }

        if need_traits_in_scope {
            // FIXME: hygiene is not considered.
            let mut doc_link_traits_in_scope = std::mem::take(&mut self.r.doc_link_traits_in_scope);
            doc_link_traits_in_scope
                .entry(self.parent_scope.module.nearest_parent_mod().expect_local())
                .or_insert_with(|| {
                    self.r
                        .traits_in_scope(None, &self.parent_scope, SyntaxContext::root(), None)
                        .into_iter()
                        .filter_map(|tr| {
                            if !tr.def_id.is_local()
                                && self.r.tcx.sess.crate_types().contains(&CrateType::ProcMacro)
                                && matches!(
                                    self.r.tcx.sess.opts.resolve_doc_links,
                                    ResolveDocLinks::ExportedMetadata
                                )
                            {
                                // Encoding foreign def ids in proc macro crate metadata will ICE.
                                return None;
                            }
                            Some(tr.def_id)
                        })
                        .collect()
                });
            self.r.doc_link_traits_in_scope = doc_link_traits_in_scope;
        }
    }
}

struct LifetimeCountVisitor<'a, 'b, 'tcx> {
    r: &'b mut Resolver<'a, 'tcx>,
}

/// Walks the whole crate in DFS order, visiting each item, counting the declared number of
/// lifetime generic parameters.
impl<'ast> Visitor<'ast> for LifetimeCountVisitor<'_, '_, '_> {
    fn visit_item(&mut self, item: &'ast Item) {
        match &item.kind {
            ItemKind::TyAlias(box TyAlias { ref generics, .. })
            | ItemKind::Fn(box Fn { ref generics, .. })
            | ItemKind::Enum(_, ref generics)
            | ItemKind::Struct(_, ref generics)
            | ItemKind::Union(_, ref generics)
            | ItemKind::Impl(box Impl { ref generics, .. })
            | ItemKind::Trait(box Trait { ref generics, .. })
            | ItemKind::TraitAlias(ref generics, _) => {
                let def_id = self.r.local_def_id(item.id);
                let count = generics
                    .params
                    .iter()
                    .filter(|param| matches!(param.kind, ast::GenericParamKind::Lifetime { .. }))
                    .count();
                self.r.item_generics_num_lifetimes.insert(def_id, count);
            }

            ItemKind::Mod(..)
            | ItemKind::ForeignMod(..)
            | ItemKind::Static(..)
            | ItemKind::Const(..)
            | ItemKind::Use(..)
            | ItemKind::ExternCrate(..)
            | ItemKind::MacroDef(..)
            | ItemKind::GlobalAsm(..)
            | ItemKind::MacCall(..) => {}
        }
        visit::walk_item(self, item)
    }
}

impl<'a, 'tcx> Resolver<'a, 'tcx> {
    pub(crate) fn late_resolve_crate(&mut self, krate: &Crate) {
        visit::walk_crate(&mut LifetimeCountVisitor { r: self }, krate);
        let mut late_resolution_visitor = LateResolutionVisitor::new(self);
        late_resolution_visitor.resolve_doc_links(&krate.attrs, MaybeExported::Ok(CRATE_NODE_ID));
        visit::walk_crate(&mut late_resolution_visitor, krate);
        for (id, span) in late_resolution_visitor.diagnostic_metadata.unused_labels.iter() {
            self.lint_buffer.buffer_lint(lint::builtin::UNUSED_LABELS, *id, *span, "unused label");
        }
    }
}
