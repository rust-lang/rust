//! "Late resolution" is the pass that resolves most of names in a crate beside imports and macros.
//! It runs when the crate is fully expanded and its module structure is fully built.
//! So it just walks through the crate and resolves all the expressions, types, etc.
//!
//! If you wonder why there's no `early.rs`, that's because it's split into three files -
//! `build_reduced_graph.rs`, `macros.rs` and `imports.rs`.

use RibKind::*;

use crate::{path_names_to_string, BindingError, CrateLint, LexicalScopeBinding};
use crate::{Module, ModuleOrUniformRoot, ParentScope, PathResult};
use crate::{ResolutionError, Resolver, Segment, UseError};

use rustc_ast::ptr::P;
use rustc_ast::visit::{self, AssocCtxt, FnCtxt, FnKind, Visitor};
use rustc_ast::*;
use rustc_ast_lowering::ResolverAstLowering;
use rustc_data_structures::fx::{FxHashMap, FxHashSet};
use rustc_errors::DiagnosticId;
use rustc_hir::def::Namespace::{self, *};
use rustc_hir::def::{self, CtorKind, DefKind, PartialRes, PerNS};
use rustc_hir::def_id::{DefId, CRATE_DEF_INDEX};
use rustc_hir::{PrimTy, TraitCandidate};
use rustc_middle::{bug, span_bug};
use rustc_session::lint;
use rustc_span::symbol::{kw, sym, Ident, Symbol};
use rustc_span::Span;
use smallvec::{smallvec, SmallVec};

use rustc_span::source_map::{respan, Spanned};
use std::collections::{hash_map::Entry, BTreeSet};
use std::mem::{replace, take};
use tracing::debug;

mod diagnostics;
crate mod lifetimes;

type Res = def::Res<NodeId>;

type IdentMap<T> = FxHashMap<Ident, T>;

/// Map from the name in a pattern to its binding mode.
type BindingMap = IdentMap<BindingInfo>;

#[derive(Copy, Clone, Debug)]
struct BindingInfo {
    span: Span,
    binding_mode: BindingMode,
}

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
enum PatternSource {
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
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
crate enum HasGenericParams {
    Yes,
    No,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
crate enum ConstantItemKind {
    Const,
    Static,
}

/// The rib kind restricts certain accesses,
/// e.g. to a `Res::Local` of an outer item.
#[derive(Copy, Clone, Debug)]
crate enum RibKind<'a> {
    /// No restriction needs to be applied.
    NormalRibKind,

    /// We passed through an impl or trait and are now in one of its
    /// methods or associated types. Allow references to ty params that impl or trait
    /// binds. Disallow any other upvars (including other ty params that are
    /// upvars).
    AssocItemRibKind,

    /// We passed through a closure. Disallow labels.
    ClosureOrAsyncRibKind,

    /// We passed through a function definition. Disallow upvars.
    /// Permit only those const parameters that are specified in the function's generics.
    FnItemRibKind,

    /// We passed through an item scope. Disallow upvars.
    ItemRibKind(HasGenericParams),

    /// We're in a constant item. Can't refer to dynamic stuff.
    ///
    /// The `bool` indicates if this constant may reference generic parameters
    /// and is used to only allow generic parameters to be used in trivial constant expressions.
    ConstantItemRibKind(bool, Option<(Ident, ConstantItemKind)>),

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
}

impl RibKind<'_> {
    /// Whether this rib kind contains generic parameters, as opposed to local
    /// variables.
    crate fn contains_params(&self) -> bool {
        match self {
            NormalRibKind
            | ClosureOrAsyncRibKind
            | FnItemRibKind
            | ConstantItemRibKind(..)
            | ModuleRibKind(_)
            | MacroDefinition(_)
            | ConstParamTyRibKind => false,
            AssocItemRibKind | ItemRibKind(_) | ForwardGenericParamBanRibKind => true,
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
/// Different [rib kinds](enum.RibKind) are transparent for different names.
///
/// The resolution keeps a separate stack of ribs as it traverses the AST for each namespace. When
/// resolving, the name is looked up from inside out.
#[derive(Debug)]
crate struct Rib<'a, R = Res> {
    pub bindings: IdentMap<R>,
    pub kind: RibKind<'a>,
}

impl<'a, R> Rib<'a, R> {
    fn new(kind: RibKind<'a>) -> Rib<'a, R> {
        Rib { bindings: Default::default(), kind }
    }
}

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
crate enum AliasPossibility {
    No,
    Maybe,
}

#[derive(Copy, Clone, Debug)]
crate enum PathSource<'a> {
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

    crate fn is_expected(self, res: Res) -> bool {
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
                    | Res::SelfTy(..)
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
                        | DefKind::Static
                        | DefKind::Fn
                        | DefKind::AssocFn
                        | DefKind::AssocConst
                        | DefKind::ConstParam,
                    _,
                ) | Res::Local(..)
                    | Res::SelfCtor(..)
            ),
            PathSource::Pat => matches!(
                res,
                Res::Def(
                    DefKind::Ctor(_, CtorKind::Const) | DefKind::Const | DefKind::AssocConst,
                    _,
                ) | Res::SelfCtor(..)
            ),
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
                ) | Res::SelfTy(..)
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

    /// If we are currently in a trait object definition. Used to point at the bounds when
    /// encountering a struct or enum.
    current_trait_object: Option<&'ast [ast::GenericBound]>,

    /// Given `where <T as Bar>::Baz: String`, suggest `where T: Bar<Baz = String>`.
    current_where_predicate: Option<&'ast WherePredicate>,
}

struct LateResolutionVisitor<'a, 'b, 'ast> {
    r: &'b mut Resolver<'a>,

    /// The module that represents the current item scope.
    parent_scope: ParentScope<'a>,

    /// The current set of local scopes for types and values.
    /// FIXME #4948: Reuse ribs to avoid allocation.
    ribs: PerNS<Vec<Rib<'a>>>,

    /// The current set of local scopes, for labels.
    label_ribs: Vec<Rib<'a, NodeId>>,

    /// The trait that the current context can refer to.
    current_trait_ref: Option<(Module<'a>, TraitRef)>,

    /// Fields used to add information to diagnostic errors.
    diagnostic_metadata: DiagnosticMetadata<'ast>,

    /// State used to know whether to ignore resolution errors for function bodies.
    ///
    /// In particular, rustdoc uses this to avoid giving errors for `cfg()` items.
    /// In most cases this will be `None`, in which case errors will always be reported.
    /// If it is `true`, then it will be updated when entering a nested function or trait body.
    in_func_body: bool,
}

/// Walks the whole crate in DFS order, visiting each item, resolving names as it goes.
impl<'a: 'ast, 'ast> Visitor<'ast> for LateResolutionVisitor<'a, '_, 'ast> {
    fn visit_attribute(&mut self, _: &'ast Attribute) {
        // We do not want to resolve expressions that appear in attributes,
        // as they do not correspond to actual code.
    }
    fn visit_item(&mut self, item: &'ast Item) {
        let prev = replace(&mut self.diagnostic_metadata.current_item, Some(item));
        // Always report errors in items we just entered.
        let old_ignore = replace(&mut self.in_func_body, false);
        self.resolve_item(item);
        self.in_func_body = old_ignore;
        self.diagnostic_metadata.current_item = prev;
    }
    fn visit_arm(&mut self, arm: &'ast Arm) {
        self.resolve_arm(arm);
    }
    fn visit_block(&mut self, block: &'ast Block) {
        self.resolve_block(block);
    }
    fn visit_anon_const(&mut self, constant: &'ast AnonConst) {
        // We deal with repeat expressions explicitly in `resolve_expr`.
        self.resolve_anon_const(constant, IsRepeatExpr::No);
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
        match ty.kind {
            TyKind::Path(ref qself, ref path) => {
                self.smart_resolve_path(ty.id, qself.as_ref(), path, PathSource::Type);
            }
            TyKind::ImplicitSelf => {
                let self_ty = Ident::with_dummy_span(kw::SelfUpper);
                let res = self
                    .resolve_ident_in_lexical_scope(self_ty, TypeNS, Some(ty.id), ty.span)
                    .map_or(Res::Err, |d| d.res());
                self.r.record_partial_res(ty.id, PartialRes::new(res));
            }
            TyKind::TraitObject(ref bounds, ..) => {
                self.diagnostic_metadata.current_trait_object = Some(&bounds[..]);
            }
            _ => (),
        }
        visit::walk_ty(self, ty);
        self.diagnostic_metadata.current_trait_object = prev;
    }
    fn visit_poly_trait_ref(&mut self, tref: &'ast PolyTraitRef, m: &'ast TraitBoundModifier) {
        self.smart_resolve_path(
            tref.trait_ref.ref_id,
            None,
            &tref.trait_ref.path,
            PathSource::Trait(AliasPossibility::Maybe),
        );
        visit::walk_poly_trait_ref(self, tref, m);
    }
    fn visit_foreign_item(&mut self, foreign_item: &'ast ForeignItem) {
        match foreign_item.kind {
            ForeignItemKind::Fn(box Fn { ref generics, .. })
            | ForeignItemKind::TyAlias(box TyAlias { ref generics, .. }) => {
                self.with_generic_param_rib(generics, ItemRibKind(HasGenericParams::Yes), |this| {
                    visit::walk_foreign_item(this, foreign_item);
                });
            }
            ForeignItemKind::Static(..) => {
                self.with_item_rib(HasGenericParams::No, |this| {
                    visit::walk_foreign_item(this, foreign_item);
                });
            }
            ForeignItemKind::MacCall(..) => {
                visit::walk_foreign_item(self, foreign_item);
            }
        }
    }
    fn visit_fn(&mut self, fn_kind: FnKind<'ast>, sp: Span, _: NodeId) {
        let rib_kind = match fn_kind {
            // Bail if there's no body.
            FnKind::Fn(.., None) => return visit::walk_fn(self, fn_kind, sp),
            FnKind::Fn(FnCtxt::Free | FnCtxt::Foreign, ..) => FnItemRibKind,
            FnKind::Fn(FnCtxt::Assoc(_), ..) => NormalRibKind,
            FnKind::Closure(..) => ClosureOrAsyncRibKind,
        };
        let previous_value = self.diagnostic_metadata.current_function;
        if matches!(fn_kind, FnKind::Fn(..)) {
            self.diagnostic_metadata.current_function = Some((fn_kind, sp));
        }
        debug!("(resolving function) entering function");
        let declaration = fn_kind.decl();

        // Create a value rib for the function.
        self.with_rib(ValueNS, rib_kind, |this| {
            // Create a label rib for the function.
            this.with_label_rib(rib_kind, |this| {
                // Add each argument to the rib.
                this.resolve_params(&declaration.inputs);

                visit::walk_fn_ret_ty(this, &declaration.output);

                // Ignore errors in function bodies if this is rustdoc
                // Be sure not to set this until the function signature has been resolved.
                let previous_state = replace(&mut this.in_func_body, true);
                // Resolve the function body, potentially inside the body of an async closure
                match fn_kind {
                    FnKind::Fn(.., body) => walk_list!(this, visit_block, body),
                    FnKind::Closure(_, body) => this.visit_expr(body),
                };

                debug!("(resolving function) leaving function");
                this.in_func_body = previous_state;
            })
        });
        self.diagnostic_metadata.current_function = previous_value;
    }

    fn visit_generics(&mut self, generics: &'ast Generics) {
        // For type parameter defaults, we have to ban access
        // to following type parameters, as the InternalSubsts can only
        // provide previous type parameters as they're built. We
        // put all the parameters on the ban list and then remove
        // them one by one as they are processed and become available.
        let mut forward_ty_ban_rib = Rib::new(ForwardGenericParamBanRibKind);
        let mut forward_const_ban_rib = Rib::new(ForwardGenericParamBanRibKind);
        for param in generics.params.iter() {
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
        if self.diagnostic_metadata.current_self_item.is_some() {
            // (`Some` if + only if we are in ADT's generics.)
            forward_ty_ban_rib.bindings.insert(Ident::with_dummy_span(kw::SelfUpper), Res::Err);
        }

        for param in &generics.params {
            match param.kind {
                GenericParamKind::Lifetime => self.visit_generic_param(param),
                GenericParamKind::Type { ref default } => {
                    for bound in &param.bounds {
                        self.visit_param_bound(bound);
                    }

                    if let Some(ref ty) = default {
                        self.ribs[TypeNS].push(forward_ty_ban_rib);
                        self.ribs[ValueNS].push(forward_const_ban_rib);
                        self.visit_ty(ty);
                        forward_const_ban_rib = self.ribs[ValueNS].pop().unwrap();
                        forward_ty_ban_rib = self.ribs[TypeNS].pop().unwrap();
                    }

                    // Allow all following defaults to refer to this type parameter.
                    forward_ty_ban_rib.bindings.remove(&Ident::with_dummy_span(param.ident.name));
                }
                GenericParamKind::Const { ref ty, kw_span: _, ref default } => {
                    // Const parameters can't have param bounds.
                    assert!(param.bounds.is_empty());

                    self.ribs[TypeNS].push(Rib::new(ConstParamTyRibKind));
                    self.ribs[ValueNS].push(Rib::new(ConstParamTyRibKind));
                    self.visit_ty(ty);
                    self.ribs[TypeNS].pop().unwrap();
                    self.ribs[ValueNS].pop().unwrap();

                    if let Some(ref expr) = default {
                        self.ribs[TypeNS].push(forward_ty_ban_rib);
                        self.ribs[ValueNS].push(forward_const_ban_rib);
                        self.visit_anon_const(expr);
                        forward_const_ban_rib = self.ribs[ValueNS].pop().unwrap();
                        forward_ty_ban_rib = self.ribs[TypeNS].pop().unwrap();
                    }

                    // Allow all following defaults to refer to this const parameter.
                    forward_const_ban_rib
                        .bindings
                        .remove(&Ident::with_dummy_span(param.ident.name));
                }
            }
        }
        for p in &generics.where_clause.predicates {
            self.visit_where_predicate(p);
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
                            self.resolve_ident_in_lexical_scope(
                                path.segments[0].ident,
                                ns,
                                None,
                                path.span,
                            )
                            .is_some()
                        };
                        if !check_ns(TypeNS) && check_ns(ValueNS) {
                            // This must be equivalent to `visit_anon_const`, but we cannot call it
                            // directly due to visitor lifetimes so we have to copy-paste some code.
                            //
                            // Note that we might not be inside of an repeat expression here,
                            // but considering that `IsRepeatExpr` is only relevant for
                            // non-trivial constants this is doesn't matter.
                            self.with_constant_rib(IsRepeatExpr::No, true, None, |this| {
                                this.smart_resolve_path(
                                    ty.id,
                                    qself.as_ref(),
                                    path,
                                    PathSource::Expr(None),
                                );

                                if let Some(ref qself) = *qself {
                                    this.visit_ty(&qself.ty);
                                }
                                this.visit_path(path, ty.id);
                            });

                            self.diagnostic_metadata.currently_processing_generics = prev;
                            return;
                        }
                    }
                }

                self.visit_ty(ty);
            }
            GenericArg::Lifetime(lt) => self.visit_lifetime(lt),
            GenericArg::Const(ct) => self.visit_anon_const(ct),
        }
        self.diagnostic_metadata.currently_processing_generics = prev;
    }

    fn visit_where_predicate(&mut self, p: &'ast WherePredicate) {
        debug!("visit_where_predicate {:?}", p);
        let previous_value =
            replace(&mut self.diagnostic_metadata.current_where_predicate, Some(p));
        visit::walk_where_predicate(self, p);
        self.diagnostic_metadata.current_where_predicate = previous_value;
    }
}

impl<'a: 'ast, 'b, 'ast> LateResolutionVisitor<'a, 'b, 'ast> {
    fn new(resolver: &'b mut Resolver<'a>) -> LateResolutionVisitor<'a, 'b, 'ast> {
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
            label_ribs: Vec::new(),
            current_trait_ref: None,
            diagnostic_metadata: DiagnosticMetadata::default(),
            // errors at module scope should always be reported
            in_func_body: false,
        }
    }

    fn resolve_ident_in_lexical_scope(
        &mut self,
        ident: Ident,
        ns: Namespace,
        record_used_id: Option<NodeId>,
        path_span: Span,
    ) -> Option<LexicalScopeBinding<'a>> {
        self.r.resolve_ident_in_lexical_scope(
            ident,
            ns,
            &self.parent_scope,
            record_used_id,
            path_span,
            &self.ribs[ns],
        )
    }

    fn resolve_path(
        &mut self,
        path: &[Segment],
        opt_ns: Option<Namespace>, // `None` indicates a module path in import
        record_used: bool,
        path_span: Span,
        crate_lint: CrateLint,
    ) -> PathResult<'a> {
        self.r.resolve_path_with_ribs(
            path,
            opt_ns,
            &self.parent_scope,
            record_used,
            path_span,
            crate_lint,
            Some(&self.ribs),
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

    /// Searches the current set of local scopes for labels. Returns the `NodeId` of the resolved
    /// label and reports an error if the label is not found or is unreachable.
    fn resolve_label(&self, mut label: Ident) -> Option<NodeId> {
        let mut suggestion = None;

        // Preserve the original span so that errors contain "in this macro invocation"
        // information.
        let original_span = label.span;

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
                return if self.is_label_valid_from_rib(i) {
                    Some(*id)
                } else {
                    self.report_error(
                        original_span,
                        ResolutionError::UnreachableLabel {
                            name: label.name,
                            definition_span: ident.span,
                            suggestion,
                        },
                    );

                    None
                };
            }

            // Diagnostics: Check if this rib contains a label with a similar name, keep track of
            // the first such label that is encountered.
            suggestion = suggestion.or_else(|| self.suggestion_for_label_in_rib(i, label));
        }

        self.report_error(
            original_span,
            ResolutionError::UndeclaredLabel { name: label.name, suggestion },
        );
        None
    }

    /// Determine whether or not a label from the `rib_index`th label rib is reachable.
    fn is_label_valid_from_rib(&self, rib_index: usize) -> bool {
        let ribs = &self.label_ribs[rib_index + 1..];

        for rib in ribs {
            match rib.kind {
                NormalRibKind | MacroDefinition(..) => {
                    // Nothing to do. Continue.
                }

                AssocItemRibKind
                | ClosureOrAsyncRibKind
                | FnItemRibKind
                | ItemRibKind(..)
                | ConstantItemRibKind(..)
                | ModuleRibKind(..)
                | ForwardGenericParamBanRibKind
                | ConstParamTyRibKind => {
                    return false;
                }
            }
        }

        true
    }

    fn resolve_adt(&mut self, item: &'ast Item, generics: &'ast Generics) {
        debug!("resolve_adt");
        self.with_current_self_item(item, |this| {
            this.with_generic_param_rib(generics, ItemRibKind(HasGenericParams::Yes), |this| {
                let item_def_id = this.r.local_def_id(item.id).to_def_id();
                this.with_self_rib(Res::SelfTy(None, Some((item_def_id, false))), |this| {
                    visit::walk_item(this, item);
                });
            });
        });
    }

    fn future_proof_import(&mut self, use_tree: &UseTree) {
        let segments = &use_tree.prefix.segments;
        if !segments.is_empty() {
            let ident = segments[0].ident;
            if ident.is_path_segment_keyword() || ident.span.rust_2015() {
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
                        .session
                        .span_err(ident.span, &format!("imports cannot refer to {}", what));
                }
            };

            for &ns in nss {
                match self.resolve_ident_in_lexical_scope(ident, ns, None, use_tree.prefix.span) {
                    Some(LexicalScopeBinding::Res(..)) => {
                        report_error(self, ns);
                    }
                    Some(LexicalScopeBinding::Item(binding)) => {
                        let orig_unusable_binding =
                            replace(&mut self.r.unusable_binding, Some(binding));
                        if let Some(LexicalScopeBinding::Res(..)) = self
                            .resolve_ident_in_lexical_scope(ident, ns, None, use_tree.prefix.span)
                        {
                            report_error(self, ns);
                        }
                        self.r.unusable_binding = orig_unusable_binding;
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
        let name = item.ident.name;
        debug!("(resolving item) resolving {} ({:?})", name, item.kind);

        match item.kind {
            ItemKind::TyAlias(box TyAlias { ref generics, .. })
            | ItemKind::Fn(box Fn { ref generics, .. }) => {
                self.compute_num_lifetime_params(item.id, generics);
                self.with_generic_param_rib(generics, ItemRibKind(HasGenericParams::Yes), |this| {
                    visit::walk_item(this, item)
                });
            }

            ItemKind::Enum(_, ref generics)
            | ItemKind::Struct(_, ref generics)
            | ItemKind::Union(_, ref generics) => {
                self.compute_num_lifetime_params(item.id, generics);
                self.resolve_adt(item, generics);
            }

            ItemKind::Impl(box Impl {
                ref generics,
                ref of_trait,
                ref self_ty,
                items: ref impl_items,
                ..
            }) => {
                self.compute_num_lifetime_params(item.id, generics);
                self.resolve_implementation(generics, of_trait, &self_ty, item.id, impl_items);
            }

            ItemKind::Trait(box Trait { ref generics, ref bounds, ref items, .. }) => {
                self.compute_num_lifetime_params(item.id, generics);
                // Create a new rib for the trait-wide type parameters.
                self.with_generic_param_rib(generics, ItemRibKind(HasGenericParams::Yes), |this| {
                    let local_def_id = this.r.local_def_id(item.id).to_def_id();
                    this.with_self_rib(Res::SelfTy(Some(local_def_id), None), |this| {
                        this.visit_generics(generics);
                        walk_list!(this, visit_param_bound, bounds);

                        let walk_assoc_item = |this: &mut Self, generics, item| {
                            this.with_generic_param_rib(generics, AssocItemRibKind, |this| {
                                visit::walk_assoc_item(this, item, AssocCtxt::Trait)
                            });
                        };

                        this.with_trait_items(items, |this| {
                            for item in items {
                                match &item.kind {
                                    AssocItemKind::Const(_, ty, default) => {
                                        this.visit_ty(ty);
                                        // Only impose the restrictions of `ConstRibKind` for an
                                        // actual constant expression in a provided default.
                                        if let Some(expr) = default {
                                            // We allow arbitrary const expressions inside of associated consts,
                                            // even if they are potentially not const evaluatable.
                                            //
                                            // Type parameters can already be used and as associated consts are
                                            // not used as part of the type system, this is far less surprising.
                                            this.with_constant_rib(
                                                IsRepeatExpr::No,
                                                true,
                                                None,
                                                |this| this.visit_expr(expr),
                                            );
                                        }
                                    }
                                    AssocItemKind::Fn(box Fn { generics, .. }) => {
                                        walk_assoc_item(this, generics, item);
                                    }
                                    AssocItemKind::TyAlias(box TyAlias { generics, .. }) => {
                                        walk_assoc_item(this, generics, item);
                                    }
                                    AssocItemKind::MacCall(_) => {
                                        panic!("unexpanded macro in resolve!")
                                    }
                                };
                            }
                        });
                    });
                });
            }

            ItemKind::TraitAlias(ref generics, ref bounds) => {
                self.compute_num_lifetime_params(item.id, generics);
                // Create a new rib for the trait-wide type parameters.
                self.with_generic_param_rib(generics, ItemRibKind(HasGenericParams::Yes), |this| {
                    let local_def_id = this.r.local_def_id(item.id).to_def_id();
                    this.with_self_rib(Res::SelfTy(Some(local_def_id), None), |this| {
                        this.visit_generics(generics);
                        walk_list!(this, visit_param_bound, bounds);
                    });
                });
            }

            ItemKind::Mod(..) | ItemKind::ForeignMod(_) => {
                self.with_scope(item.id, |this| {
                    visit::walk_item(this, item);
                });
            }

            ItemKind::Static(ref ty, _, ref expr) | ItemKind::Const(_, ref ty, ref expr) => {
                self.with_item_rib(HasGenericParams::No, |this| {
                    this.visit_ty(ty);
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
                            true,
                            Some((item.ident, constant_item_kind)),
                            |this| this.visit_expr(expr),
                        );
                    }
                });
            }

            ItemKind::Use(ref use_tree) => {
                self.future_proof_import(use_tree);
            }

            ItemKind::ExternCrate(..) | ItemKind::MacroDef(..) => {
                // do nothing, these are just around to be encoded
            }

            ItemKind::GlobalAsm(_) => {
                visit::walk_item(self, item);
            }

            ItemKind::MacCall(_) => panic!("unexpanded macro in resolve!"),
        }
    }

    fn with_generic_param_rib<'c, F>(&'c mut self, generics: &'c Generics, kind: RibKind<'a>, f: F)
    where
        F: FnOnce(&mut Self),
    {
        debug!("with_generic_param_rib");
        let mut function_type_rib = Rib::new(kind);
        let mut function_value_rib = Rib::new(kind);
        let mut seen_bindings = FxHashMap::default();

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

        for param in &generics.params {
            if let GenericParamKind::Lifetime { .. } = param.kind {
                continue;
            }

            let ident = param.ident.normalize_to_macros_2_0();
            debug!("with_generic_param_rib: {}", param.id);

            match seen_bindings.entry(ident) {
                Entry::Occupied(entry) => {
                    let span = *entry.get();
                    let err = ResolutionError::NameAlreadyUsedInParameterList(ident.name, span);
                    self.report_error(param.ident.span, err);
                }
                Entry::Vacant(entry) => {
                    entry.insert(param.ident.span);
                }
            }

            // Plain insert (no renaming).
            let (rib, def_kind) = match param.kind {
                GenericParamKind::Type { .. } => (&mut function_type_rib, DefKind::TyParam),
                GenericParamKind::Const { .. } => (&mut function_value_rib, DefKind::ConstParam),
                _ => unreachable!(),
            };
            let res = Res::Def(def_kind, self.r.local_def_id(param.id).to_def_id());
            self.r.record_partial_res(param.id, PartialRes::new(res));
            rib.bindings.insert(ident, res);
        }

        self.ribs[ValueNS].push(function_value_rib);
        self.ribs[TypeNS].push(function_type_rib);

        f(self);

        self.ribs[TypeNS].pop();
        self.ribs[ValueNS].pop();
    }

    fn with_label_rib(&mut self, kind: RibKind<'a>, f: impl FnOnce(&mut Self)) {
        self.label_ribs.push(Rib::new(kind));
        f(self);
        self.label_ribs.pop();
    }

    fn with_item_rib(&mut self, has_generic_params: HasGenericParams, f: impl FnOnce(&mut Self)) {
        let kind = ItemRibKind(has_generic_params);
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
    fn with_constant_rib(
        &mut self,
        is_repeat: IsRepeatExpr,
        is_trivial: bool,
        item: Option<(Ident, ConstantItemKind)>,
        f: impl FnOnce(&mut Self),
    ) {
        debug!("with_constant_rib: is_repeat={:?} is_trivial={}", is_repeat, is_trivial);
        self.with_rib(ValueNS, ConstantItemRibKind(is_trivial, item), |this| {
            this.with_rib(
                TypeNS,
                ConstantItemRibKind(is_repeat == IsRepeatExpr::Yes || is_trivial, item),
                |this| {
                    this.with_label_rib(ConstantItemRibKind(is_trivial, item), f);
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
    fn with_trait_items<T>(
        &mut self,
        trait_items: &'ast [P<AssocItem>],
        f: impl FnOnce(&mut Self) -> T,
    ) -> T {
        let trait_assoc_items =
            replace(&mut self.diagnostic_metadata.current_trait_assoc_items, Some(&trait_items));
        let result = f(self);
        self.diagnostic_metadata.current_trait_assoc_items = trait_assoc_items;
        result
    }

    /// This is called to resolve a trait reference from an `impl` (i.e., `impl Trait for Foo`).
    fn with_optional_trait_ref<T>(
        &mut self,
        opt_trait_ref: Option<&TraitRef>,
        f: impl FnOnce(&mut Self, Option<DefId>) -> T,
    ) -> T {
        let mut new_val = None;
        let mut new_id = None;
        if let Some(trait_ref) = opt_trait_ref {
            let path: Vec<_> = Segment::from_path(&trait_ref.path);
            let res = self.smart_resolve_path_fragment(
                trait_ref.ref_id,
                None,
                &path,
                trait_ref.path.span,
                PathSource::Trait(AliasPossibility::No),
                CrateLint::SimplePath(trait_ref.ref_id),
            );
            let res = res.base_res();
            if res != Res::Err {
                new_id = Some(res.def_id());
                let span = trait_ref.path.span;
                if let PathResult::Module(ModuleOrUniformRoot::Module(module)) = self.resolve_path(
                    &path,
                    Some(TypeNS),
                    false,
                    span,
                    CrateLint::SimplePath(trait_ref.ref_id),
                ) {
                    new_val = Some((module, trait_ref.clone()));
                }
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
        generics: &'ast Generics,
        opt_trait_reference: &'ast Option<TraitRef>,
        self_type: &'ast Ty,
        item_id: NodeId,
        impl_items: &'ast [P<AssocItem>],
    ) {
        debug!("resolve_implementation");
        // If applicable, create a rib for the type parameters.
        self.with_generic_param_rib(generics, ItemRibKind(HasGenericParams::Yes), |this| {
            // Dummy self type for better errors if `Self` is used in the trait path.
            this.with_self_rib(Res::SelfTy(None, None), |this| {
                // Resolve the trait reference, if necessary.
                this.with_optional_trait_ref(opt_trait_reference.as_ref(), |this, trait_id| {
                    let item_def_id = this.r.local_def_id(item_id);

                    // Register the trait definitions from here.
                    if let Some(trait_id) = trait_id {
                        this.r.trait_impls.entry(trait_id).or_default().push(item_def_id);
                    }

                    let item_def_id = item_def_id.to_def_id();
                    this.with_self_rib(Res::SelfTy(trait_id, Some((item_def_id, false))), |this| {
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
                                for item in impl_items {
                                    use crate::ResolutionError::*;
                                    match &item.kind {
                                        AssocItemKind::Const(_default, _ty, _expr) => {
                                            debug!("resolve_implementation AssocItemKind::Const");
                                            // If this is a trait impl, ensure the const
                                            // exists in trait
                                            this.check_trait_item(
                                                item.ident,
                                                &item.kind,
                                                ValueNS,
                                                item.span,
                                                |i, s, c| ConstNotMemberOfTrait(i, s, c),
                                            );

                                            // We allow arbitrary const expressions inside of associated consts,
                                            // even if they are potentially not const evaluatable.
                                            //
                                            // Type parameters can already be used and as associated consts are
                                            // not used as part of the type system, this is far less surprising.
                                            this.with_constant_rib(
                                                IsRepeatExpr::No,
                                                true,
                                                None,
                                                |this| {
                                                    visit::walk_assoc_item(
                                                        this,
                                                        item,
                                                        AssocCtxt::Impl,
                                                    )
                                                },
                                            );
                                        }
                                        AssocItemKind::Fn(box Fn { generics, .. }) => {
                                            debug!("resolve_implementation AssocItemKind::Fn");
                                            // We also need a new scope for the impl item type parameters.
                                            this.with_generic_param_rib(
                                                generics,
                                                AssocItemRibKind,
                                                |this| {
                                                    // If this is a trait impl, ensure the method
                                                    // exists in trait
                                                    this.check_trait_item(
                                                        item.ident,
                                                        &item.kind,
                                                        ValueNS,
                                                        item.span,
                                                        |i, s, c| MethodNotMemberOfTrait(i, s, c),
                                                    );

                                                    visit::walk_assoc_item(
                                                        this,
                                                        item,
                                                        AssocCtxt::Impl,
                                                    )
                                                },
                                            );
                                        }
                                        AssocItemKind::TyAlias(box TyAlias {
                                            generics, ..
                                        }) => {
                                            debug!("resolve_implementation AssocItemKind::TyAlias");
                                            // We also need a new scope for the impl item type parameters.
                                            this.with_generic_param_rib(
                                                generics,
                                                AssocItemRibKind,
                                                |this| {
                                                    // If this is a trait impl, ensure the type
                                                    // exists in trait
                                                    this.check_trait_item(
                                                        item.ident,
                                                        &item.kind,
                                                        TypeNS,
                                                        item.span,
                                                        |i, s, c| TypeNotMemberOfTrait(i, s, c),
                                                    );

                                                    visit::walk_assoc_item(
                                                        this,
                                                        item,
                                                        AssocCtxt::Impl,
                                                    )
                                                },
                                            );
                                        }
                                        AssocItemKind::MacCall(_) => {
                                            panic!("unexpanded macro in resolve!")
                                        }
                                    }
                                }
                            });
                        });
                    });
                });
            });
        });
    }

    fn check_trait_item<F>(
        &mut self,
        ident: Ident,
        kind: &AssocItemKind,
        ns: Namespace,
        span: Span,
        err: F,
    ) where
        F: FnOnce(Ident, &str, Option<Symbol>) -> ResolutionError<'_>,
    {
        // If there is a TraitRef in scope for an impl, then the method must be in the
        // trait.
        if let Some((module, _)) = self.current_trait_ref {
            if self
                .r
                .resolve_ident_in_module(
                    ModuleOrUniformRoot::Module(module),
                    ident,
                    ns,
                    &self.parent_scope,
                    false,
                    span,
                )
                .is_err()
            {
                let candidate = self.find_similarly_named_assoc_item(ident.name, kind);
                let path = &self.current_trait_ref.as_ref().unwrap().1.path;
                self.report_error(span, err(ident, &path_names_to_string(path), candidate));
            }
        }
    }

    fn resolve_params(&mut self, params: &'ast [Param]) {
        let mut bindings = smallvec![(PatBoundCtx::Product, Default::default())];
        for Param { pat, ty, .. } in params {
            self.resolve_pattern(pat, PatternSource::FnParam, &mut bindings);
            self.visit_ty(ty);
            debug!("(resolving function / closure) recorded parameter");
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
                PatKind::Ident(binding_mode, ident, ref sub_pat)
                    if sub_pat.is_some() || self.is_base_res_local(pat.id) =>
                {
                    binding_map.insert(ident, BindingInfo { span: ident.span, binding_mode });
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
        matches!(self.r.partial_res_map.get(&nid).map(|res| res.base_res()), Some(Res::Local(..)))
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
                        if binding_outer.binding_mode != binding_inner.binding_mode {
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
        let mut missing_vars = missing_vars.iter_mut().collect::<Vec<_>>();
        missing_vars.sort_by_key(|(sym, _err)| sym.as_str());

        for (name, mut v) in missing_vars {
            if inconsistent_vars.contains_key(name) {
                v.could_be_path = false;
            }
            self.report_error(
                *v.origin.iter().next().unwrap(),
                ResolutionError::VariableNotBoundInPattern(v),
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
                        .try_resolve_as_non_binding(pat_src, pat, bmode, ident, has_sub)
                        .unwrap_or_else(|| self.fresh_binding(ident, pat.id, pat_src, bindings));
                    self.r.record_partial_res(pat.id, PartialRes::new(res));
                    self.r.record_pat_span(pat.id, pat.span);
                }
                PatKind::TupleStruct(ref qself, ref path, ref sub_patterns) => {
                    self.smart_resolve_path(
                        pat.id,
                        qself.as_ref(),
                        path,
                        PathSource::TupleStruct(
                            pat.span,
                            self.r.arenas.alloc_pattern_spans(sub_patterns.iter().map(|p| p.span)),
                        ),
                    );
                }
                PatKind::Path(ref qself, ref path) => {
                    self.smart_resolve_path(pat.id, qself.as_ref(), path, PathSource::Pat);
                }
                PatKind::Struct(ref qself, ref path, ..) => {
                    self.smart_resolve_path(pat.id, qself.as_ref(), path, PathSource::Struct);
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
        pat: &Pat,
        bm: BindingMode,
        ident: Ident,
        has_sub: bool,
    ) -> Option<Res> {
        // An immutable (no `mut`) by-value (no `ref`) binding pattern without
        // a sub pattern (no `@ $pat`) is syntactically ambiguous as it could
        // also be interpreted as a path to e.g. a constant, variant, etc.
        let is_syntactic_ambiguity = !has_sub && bm == BindingMode::ByValue(Mutability::Not);

        let ls_binding = self.resolve_ident_in_lexical_scope(ident, ValueNS, None, pat.span)?;
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
            Res::Def(DefKind::Ctor(..) | DefKind::Const | DefKind::Static, _) => {
                // This is unambiguously a fresh binding, either syntactically
                // (e.g., `IDENT @ PAT` or `ref IDENT`) or because `IDENT` resolves
                // to something unusable as a pattern (e.g., constructor function),
                // but we still conservatively report an error, see
                // issues/33118#issuecomment-233962221 for one reason why.
                let binding = binding.expect("no binding for a ctor or static");
                self.report_error(
                    ident.span,
                    ResolutionError::BindingShadowsSomethingUnacceptable {
                        shadowing_binding_descr: pat_src.descr(),
                        name: ident.name,
                        participle: if binding.is_import() { "imported" } else { "defined" },
                        article: binding.res().article(),
                        shadowed_binding_descr: binding.res().descr(),
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
                        shadowing_binding_descr: pat_src.descr(),
                        name: ident.name,
                        participle: "defined",
                        article: res.article(),
                        shadowed_binding_descr: res.descr(),
                        shadowed_binding_span: self.r.opt_span(def_id).expect("const parameter defined outside of local crate"),
                    }
                );
                None
            }
            Res::Def(DefKind::Fn, _) | Res::Local(..) | Res::Err => {
                // These entities are explicitly allowed to be shadowed by fresh bindings.
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
        qself: Option<&QSelf>,
        path: &Path,
        source: PathSource<'ast>,
    ) {
        self.smart_resolve_path_fragment(
            id,
            qself,
            &Segment::from_path(path),
            path.span,
            source,
            CrateLint::SimplePath(id),
        );
    }

    fn smart_resolve_path_fragment(
        &mut self,
        id: NodeId,
        qself: Option<&QSelf>,
        path: &[Segment],
        span: Span,
        source: PathSource<'ast>,
        crate_lint: CrateLint,
    ) -> PartialRes {
        tracing::debug!(
            "smart_resolve_path_fragment(id={:?}, qself={:?}, path={:?})",
            id,
            qself,
            path
        );
        let ns = source.namespace();

        let report_errors = |this: &mut Self, res: Option<Res>| {
            if this.should_report_errs() {
                let (err, candidates) = this.smart_resolve_report_errors(path, span, source, res);

                let def_id = this.parent_scope.module.nearest_parent_mod();
                let instead = res.is_some();
                let suggestion =
                    if res.is_none() { this.report_missing_type_error(path) } else { None };
                // get_from_node_id

                this.r.use_injections.push(UseError {
                    err,
                    candidates,
                    def_id,
                    instead,
                    suggestion,
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
            let path = match path.split_last() {
                Some((_, path)) if !path.is_empty() => path,
                _ => return Some(parent_err),
            };

            let (mut err, candidates) =
                this.smart_resolve_report_errors(path, span, PathSource::Type, None);

            if candidates.is_empty() {
                err.cancel();
                return Some(parent_err);
            }

            // There are two different error messages user might receive at
            // this point:
            // - E0412 cannot find type `{}` in this scope
            // - E0433 failed to resolve: use of undeclared type or module `{}`
            //
            // The first one is emitted for paths in type-position, and the
            // latter one - for paths in expression-position.
            //
            // Thus (since we're in expression-position at this point), not to
            // confuse the user, we want to keep the *message* from E0432 (so
            // `parent_err`), but we want *hints* from E0412 (so `err`).
            //
            // And that's what happens below - we're just mixing both messages
            // into a single one.
            let mut parent_err = this.r.into_struct_error(parent_err.span, parent_err.node);

            parent_err.cancel();

            err.message = take(&mut parent_err.message);
            err.code = take(&mut parent_err.code);
            err.children = take(&mut parent_err.children);

            drop(parent_err);

            let def_id = this.parent_scope.module.nearest_parent_mod();

            if this.should_report_errs() {
                this.r.use_injections.push(UseError {
                    err,
                    candidates,
                    def_id,
                    instead: false,
                    suggestion: None,
                });
            } else {
                err.cancel();
            }

            // We don't return `Some(parent_err)` here, because the error will
            // be already printed as part of the `use` injections
            None
        };

        let partial_res = match self.resolve_qpath_anywhere(
            id,
            qself,
            path,
            ns,
            span,
            source.defer_to_typeck(),
            crate_lint,
        ) {
            Ok(Some(partial_res)) if partial_res.unresolved_segments() == 0 => {
                if source.is_expected(partial_res.base_res()) || partial_res.base_res() == Res::Err
                {
                    partial_res
                } else {
                    report_errors(self, Some(partial_res.base_res()))
                }
            }

            Ok(Some(partial_res)) if source.defer_to_typeck() => {
                // Not fully resolved associated item `T::A::B` or `<T as Tr>::A::B`
                // or `<T>::A::B`. If `B` should be resolved in value namespace then
                // it needs to be added to the trait map.
                if ns == ValueNS {
                    let item_name = path.last().unwrap().ident;
                    let traits = self.traits_in_scope(item_name, ns);
                    self.r.trait_map.insert(id, traits);
                }

                if PrimTy::from_name(path[0].ident.name).is_some() {
                    let mut std_path = Vec::with_capacity(1 + path.len());

                    std_path.push(Segment::from_ident(Ident::with_dummy_span(sym::std)));
                    std_path.extend(path);
                    if let PathResult::Module(_) | PathResult::NonModule(_) =
                        self.resolve_path(&std_path, Some(ns), false, span, CrateLint::No)
                    {
                        // Check if we wrote `str::from_utf8` instead of `std::str::from_utf8`
                        let item_span =
                            path.iter().last().map_or(span, |segment| segment.ident.span);

                        self.r.confused_type_with_std_module.insert(item_span, span);
                        self.r.confused_type_with_std_module.insert(span, span);
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
            self.r.record_partial_res(id, partial_res);
        }

        partial_res
    }

    fn self_type_is_available(&mut self, span: Span) -> bool {
        let binding = self.resolve_ident_in_lexical_scope(
            Ident::with_dummy_span(kw::SelfUpper),
            TypeNS,
            None,
            span,
        );
        if let Some(LexicalScopeBinding::Res(res)) = binding { res != Res::Err } else { false }
    }

    fn self_value_is_available(&mut self, self_span: Span, path_span: Span) -> bool {
        let ident = Ident::new(kw::SelfLower, self_span);
        let binding = self.resolve_ident_in_lexical_scope(ident, ValueNS, None, path_span);
        if let Some(LexicalScopeBinding::Res(res)) = binding { res != Res::Err } else { false }
    }

    /// A wrapper around [`Resolver::report_error`].
    ///
    /// This doesn't emit errors for function bodies if this is rustdoc.
    fn report_error(&self, span: Span, resolution_error: ResolutionError<'_>) {
        if self.should_report_errs() {
            self.r.report_error(span, resolution_error);
        }
    }

    #[inline]
    /// If we're actually rustdoc then avoid giving a name resolution error for `cfg()` items.
    fn should_report_errs(&self) -> bool {
        !(self.r.session.opts.actually_rustdoc && self.in_func_body)
    }

    // Resolve in alternative namespaces if resolution in the primary namespace fails.
    fn resolve_qpath_anywhere(
        &mut self,
        id: NodeId,
        qself: Option<&QSelf>,
        path: &[Segment],
        primary_ns: Namespace,
        span: Span,
        defer_to_typeck: bool,
        crate_lint: CrateLint,
    ) -> Result<Option<PartialRes>, Spanned<ResolutionError<'a>>> {
        let mut fin_res = None;

        for (i, &ns) in [primary_ns, TypeNS, ValueNS].iter().enumerate() {
            if i == 0 || ns != primary_ns {
                match self.resolve_qpath(id, qself, path, ns, span, crate_lint)? {
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
        id: NodeId,
        qself: Option<&QSelf>,
        path: &[Segment],
        ns: Namespace,
        span: Span,
        crate_lint: CrateLint,
    ) -> Result<Option<PartialRes>, Spanned<ResolutionError<'a>>> {
        debug!(
            "resolve_qpath(id={:?}, qself={:?}, path={:?}, ns={:?}, span={:?})",
            id, qself, path, ns, span,
        );

        if let Some(qself) = qself {
            if qself.position == 0 {
                // This is a case like `<T>::B`, where there is no
                // trait to resolve.  In that case, we leave the `B`
                // segment to be resolved by type-check.
                return Ok(Some(PartialRes::with_unresolved_segments(
                    Res::Def(DefKind::Mod, DefId::local(CRATE_DEF_INDEX)),
                    path.len(),
                )));
            }

            // Make sure `A::B` in `<T as A::B>::C` is a trait item.
            //
            // Currently, `path` names the full item (`A::B::C`, in
            // our example).  so we extract the prefix of that that is
            // the trait (the slice upto and including
            // `qself.position`). And then we recursively resolve that,
            // but with `qself` set to `None`.
            //
            // However, setting `qself` to none (but not changing the
            // span) loses the information about where this path
            // *actually* appears, so for the purposes of the crate
            // lint we pass along information that this is the trait
            // name from a fully qualified path, and this also
            // contains the full span (the `CrateLint::QPathTrait`).
            let ns = if qself.position + 1 == path.len() { ns } else { TypeNS };
            let partial_res = self.smart_resolve_path_fragment(
                id,
                None,
                &path[..=qself.position],
                span,
                PathSource::TraitItem(ns),
                CrateLint::QPathTrait { qpath_id: id, qpath_span: qself.path_span },
            );

            // The remaining segments (the `C` in our example) will
            // have to be resolved by type-check, since that requires doing
            // trait resolution.
            return Ok(Some(PartialRes::with_unresolved_segments(
                partial_res.base_res(),
                partial_res.unresolved_segments() + path.len() - qself.position - 1,
            )));
        }

        let result = match self.resolve_path(&path, Some(ns), true, span, crate_lint) {
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
            && result.base_res() != Res::Err
            && path[0].ident.name != kw::PathRoot
            && path[0].ident.name != kw::DollarCrate
        {
            let unqualified_result = {
                match self.resolve_path(
                    &[*path.last().unwrap()],
                    Some(ns),
                    false,
                    span,
                    CrateLint::No,
                ) {
                    PathResult::NonModule(path_res) => path_res.base_res(),
                    PathResult::Module(ModuleOrUniformRoot::Module(module)) => {
                        module.res().unwrap()
                    }
                    _ => return Ok(Some(result)),
                }
            };
            if result.base_res() == unqualified_result {
                let lint = lint::builtin::UNUSED_QUALIFICATIONS;
                self.r.lint_buffer.buffer_lint(lint, id, span, "unnecessary qualification")
            }
        }

        Ok(Some(result))
    }

    fn with_resolved_label(&mut self, label: Option<Label>, id: NodeId, f: impl FnOnce(&mut Self)) {
        if let Some(label) = label {
            if label.ident.as_str().as_bytes()[1] != b'_' {
                self.diagnostic_metadata.unused_labels.insert(id, label.ident.span);
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
        {
            if let ExprKind::Type(..) = expr.kind {
                self.diagnostic_metadata.current_block_could_be_bare_struct_literal =
                    Some(block.span);
            }
        }
        // Descend into the block.
        for stmt in &block.stmts {
            if let StmtKind::Item(ref item) = stmt.kind {
                if let ItemKind::MacroDef(..) = item.kind {
                    num_macro_definition_ribs += 1;
                    let res = self.r.local_def_id(item.id).to_def_id();
                    self.ribs[ValueNS].push(Rib::new(MacroDefinition(res)));
                    self.label_ribs.push(Rib::new(MacroDefinition(res)));
                }
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
        self.ribs[ValueNS].pop();
        if anonymous_module.is_some() {
            self.ribs[TypeNS].pop();
        }
        debug!("(resolving block) leaving block");
    }

    fn resolve_anon_const(&mut self, constant: &'ast AnonConst, is_repeat: IsRepeatExpr) {
        debug!("resolve_anon_const {:?} is_repeat: {:?}", constant, is_repeat);
        self.with_constant_rib(
            is_repeat,
            constant.value.is_potential_trivial_const_param(),
            None,
            |this| {
                visit::walk_anon_const(this, constant);
            },
        );
    }

    fn resolve_expr(&mut self, expr: &'ast Expr, parent: Option<&'ast Expr>) {
        // First, record candidate traits for this expression if it could
        // result in the invocation of a method call.

        self.record_candidate_traits_for_expr_if_necessary(expr);

        // Next, resolve the node.
        match expr.kind {
            ExprKind::Path(ref qself, ref path) => {
                self.smart_resolve_path(expr.id, qself.as_ref(), path, PathSource::Expr(parent));
                visit::walk_expr(self, expr);
            }

            ExprKind::Struct(ref se) => {
                self.smart_resolve_path(expr.id, se.qself.as_ref(), &se.path, PathSource::Struct);
                visit::walk_expr(self, expr);
            }

            ExprKind::Break(Some(label), _) | ExprKind::Continue(Some(label)) => {
                if let Some(node_id) = self.resolve_label(label.ident) {
                    // Since this res is a label, it is never read.
                    self.r.label_res_map.insert(expr.id, node_id);
                    self.diagnostic_metadata.unused_labels.remove(&node_id);
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

            ExprKind::Loop(ref block, label) => self.resolve_labeled_block(label, expr.id, &block),

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
            ExprKind::MethodCall(ref segment, ref arguments, _) => {
                let mut arguments = arguments.iter();
                self.resolve_expr(arguments.next().unwrap(), Some(expr));
                for argument in arguments {
                    self.resolve_expr(argument, None);
                }
                self.visit_path_segment(expr.span, segment);
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
                            argument.is_potential_trivial_const_param(),
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
            // `async |x| ...` gets desugared to `|x| future_from_generator(|| ...)`, so we need to
            // resolve the arguments within the proper scopes so that usages of them inside the
            // closure are detected as upvars rather than normal closure arg usages.
            ExprKind::Closure(_, Async::Yes { .. }, _, ref fn_decl, ref body, _span) => {
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
            ExprKind::Async(..) | ExprKind::Closure(..) => {
                self.with_label_rib(ClosureOrAsyncRibKind, |this| visit::walk_expr(this, expr));
            }
            ExprKind::Repeat(ref elem, ref ct) => {
                self.visit_expr(elem);
                self.resolve_anon_const(ct, IsRepeatExpr::Yes);
            }
            _ => {
                visit::walk_expr(self, expr);
            }
        }
    }

    fn record_candidate_traits_for_expr_if_necessary(&mut self, expr: &'ast Expr) {
        match expr.kind {
            ExprKind::Field(_, ident) => {
                // FIXME(#6890): Even though you can't treat a method like a
                // field, we need to add any trait methods we find that match
                // the field name so that we can do some nice error reporting
                // later on in typeck.
                let traits = self.traits_in_scope(ident, ValueNS);
                self.r.trait_map.insert(expr.id, traits);
            }
            ExprKind::MethodCall(ref segment, ..) => {
                debug!("(recording candidate traits for expr) recording traits for {}", expr.id);
                let traits = self.traits_in_scope(segment.ident, ValueNS);
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

    fn compute_num_lifetime_params(&mut self, id: NodeId, generics: &Generics) {
        let def_id = self.r.local_def_id(id);
        let count = generics
            .params
            .iter()
            .filter(|param| matches!(param.kind, ast::GenericParamKind::Lifetime { .. }))
            .count();
        self.r.item_generics_num_lifetimes.insert(def_id, count);
    }
}

impl<'a> Resolver<'a> {
    pub(crate) fn late_resolve_crate(&mut self, krate: &Crate) {
        let mut late_resolution_visitor = LateResolutionVisitor::new(self);
        visit::walk_crate(&mut late_resolution_visitor, krate);
        for (id, span) in late_resolution_visitor.diagnostic_metadata.unused_labels.iter() {
            self.lint_buffer.buffer_lint(lint::builtin::UNUSED_LABELS, *id, *span, "unused label");
        }
    }
}
