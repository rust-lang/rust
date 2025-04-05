// ignore-tidy-filelength
//! "Late resolution" is the pass that resolves most of names in a crate beside imports and macros.
//! It runs when the crate is fully expanded and its module structure is fully built.
//! So it just walks through the crate and resolves all the expressions, types, etc.
//!
//! If you wonder why there's no `early.rs`, that's because it's split into three files -
//! `build_reduced_graph.rs`, `macros.rs` and `imports.rs`.

use std::assert_matches::debug_assert_matches;
use std::borrow::Cow;
use std::collections::BTreeSet;
use std::collections::hash_map::Entry;
use std::mem::{replace, swap, take};

use rustc_ast::ptr::P;
use rustc_ast::visit::{
    AssocCtxt, BoundKind, FnCtxt, FnKind, Visitor, try_visit, visit_opt, walk_list,
};
use rustc_ast::*;
use rustc_data_structures::fx::{FxHashMap, FxHashSet, FxIndexMap};
use rustc_data_structures::unord::{UnordMap, UnordSet};
use rustc_errors::codes::*;
use rustc_errors::{
    Applicability, DiagArgValue, ErrorGuaranteed, IntoDiagArg, StashKey, Suggestions,
};
use rustc_hir::def::Namespace::{self, *};
use rustc_hir::def::{self, CtorKind, DefKind, LifetimeRes, NonMacroAttrKind, PartialRes, PerNS};
use rustc_hir::def_id::{CRATE_DEF_ID, DefId, LOCAL_CRATE, LocalDefId};
use rustc_hir::{MissingLifetimeKind, PrimTy, TraitCandidate};
use rustc_middle::middle::resolve_bound_vars::Set1;
use rustc_middle::ty::DelegationFnSig;
use rustc_middle::{bug, span_bug};
use rustc_session::config::{CrateType, ResolveDocLinks};
use rustc_session::lint::{self, BuiltinLintDiag};
use rustc_session::parse::feature_err;
use rustc_span::source_map::{Spanned, respan};
use rustc_span::{BytePos, Ident, Span, Symbol, SyntaxContext, kw, sym};
use smallvec::{SmallVec, smallvec};
use thin_vec::ThinVec;
use tracing::{debug, instrument, trace};

use crate::{
    BindingError, BindingKey, Finalize, LexicalScopeBinding, Module, ModuleOrUniformRoot,
    NameBinding, ParentScope, PathResult, ResolutionError, Resolver, Segment, TyCtxt, UseError,
    Used, errors, path_names_to_string, rustdoc,
};

mod diagnostics;

type Res = def::Res<NodeId>;

use diagnostics::{ElisionFnParameter, LifetimeElisionCandidate, MissingLifetime};

#[derive(Copy, Clone, Debug)]
struct BindingInfo {
    span: Span,
    annotation: BindingMode,
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

struct IsNeverPattern;

/// Describes whether an `AnonConst` is a type level const arg or
/// some other form of anon const (i.e. inline consts or enum discriminants)
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum AnonConstKind {
    EnumDiscriminant,
    FieldDefaultValue,
    InlineConst,
    ConstArg(IsRepeatExpr),
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

impl IntoDiagArg for PatternSource {
    fn into_diag_arg(self, _: &mut Option<std::path::PathBuf>) -> DiagArgValue {
        DiagArgValue::Str(Cow::Borrowed(self.descr()))
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
    No(NoConstantGenericsReason),
}

impl ConstantHasGenerics {
    fn force_yes_if(self, b: bool) -> Self {
        if b { Self::Yes } else { self }
    }
}

/// Reason for why an anon const is not allowed to reference generic parameters
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub(crate) enum NoConstantGenericsReason {
    /// Const arguments are only allowed to use generic parameters when:
    /// - `feature(generic_const_exprs)` is enabled
    /// or
    /// - the const argument is a sole const generic parameter, i.e. `foo::<{ N }>()`
    ///
    /// If neither of the above are true then this is used as the cause.
    NonTrivialConstArg,
    /// Enum discriminants are not allowed to reference generic parameters ever, this
    /// is used when an anon const is in the following position:
    ///
    /// ```rust,compile_fail
    /// enum Foo<const N: isize> {
    ///     Variant = { N }, // this anon const is not allowed to use generics
    /// }
    /// ```
    IsEnumDiscriminant,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub(crate) enum ConstantItemKind {
    Const,
    Static,
}

impl ConstantItemKind {
    pub(crate) fn as_str(&self) -> &'static str {
        match self {
            Self::Const => "const",
            Self::Static => "static",
        }
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
enum RecordPartialRes {
    Yes,
    No,
}

/// The rib kind restricts certain accesses,
/// e.g. to a `Res::Local` of an outer item.
#[derive(Copy, Clone, Debug)]
pub(crate) enum RibKind<'ra> {
    /// No restriction needs to be applied.
    Normal,

    /// We passed through an impl or trait and are now in one of its
    /// methods or associated types. Allow references to ty params that impl or trait
    /// binds. Disallow any other upvars (including other ty params that are
    /// upvars).
    AssocItem,

    /// We passed through a function, closure or coroutine signature. Disallow labels.
    FnOrCoroutine,

    /// We passed through an item scope. Disallow upvars.
    Item(HasGenericParams, DefKind),

    /// We're in a constant item. Can't refer to dynamic stuff.
    ///
    /// The item may reference generic parameters in trivial constant expressions.
    /// All other constants aren't allowed to use generic params at all.
    ConstantItem(ConstantHasGenerics, Option<(Ident, ConstantItemKind)>),

    /// We passed through a module.
    Module(Module<'ra>),

    /// We passed through a `macro_rules!` statement
    MacroDefinition(DefId),

    /// All bindings in this rib are generic parameters that can't be used
    /// from the default of a generic parameter because they're not declared
    /// before said generic parameter. Also see the `visit_generics` override.
    ForwardGenericParamBan(ForwardGenericParamBanReason),

    /// We are inside of the type of a const parameter. Can't refer to any
    /// parameters.
    ConstParamTy,

    /// We are inside a `sym` inline assembly operand. Can only refer to
    /// globals.
    InlineAsmSym,
}

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub(crate) enum ForwardGenericParamBanReason {
    Default,
    ConstParamTy,
}

impl RibKind<'_> {
    /// Whether this rib kind contains generic parameters, as opposed to local
    /// variables.
    pub(crate) fn contains_params(&self) -> bool {
        match self {
            RibKind::Normal
            | RibKind::FnOrCoroutine
            | RibKind::ConstantItem(..)
            | RibKind::Module(_)
            | RibKind::MacroDefinition(_)
            | RibKind::InlineAsmSym => false,
            RibKind::ConstParamTy
            | RibKind::AssocItem
            | RibKind::Item(..)
            | RibKind::ForwardGenericParamBan(_) => true,
        }
    }

    /// This rib forbids referring to labels defined in upwards ribs.
    fn is_label_barrier(self) -> bool {
        match self {
            RibKind::Normal | RibKind::MacroDefinition(..) => false,

            RibKind::AssocItem
            | RibKind::FnOrCoroutine
            | RibKind::Item(..)
            | RibKind::ConstantItem(..)
            | RibKind::Module(..)
            | RibKind::ForwardGenericParamBan(_)
            | RibKind::ConstParamTy
            | RibKind::InlineAsmSym => true,
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
pub(crate) struct Rib<'ra, R = Res> {
    pub bindings: FxIndexMap<Ident, R>,
    pub patterns_with_skipped_bindings: UnordMap<DefId, Vec<(Span, Result<(), ErrorGuaranteed>)>>,
    pub kind: RibKind<'ra>,
}

impl<'ra, R> Rib<'ra, R> {
    fn new(kind: RibKind<'ra>) -> Rib<'ra, R> {
        Rib {
            bindings: Default::default(),
            patterns_with_skipped_bindings: Default::default(),
            kind,
        }
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

    /// Resolves elided lifetimes to `'static` if there are no other lifetimes in scope,
    /// otherwise give a warning that the previous behavior of introducing a new early-bound
    /// lifetime is a bug and will be removed (if `emit_lint` is enabled).
    StaticIfNoLifetimeInScope { lint_id: NodeId, emit_lint: bool },

    /// Signal we cannot find which should be the anonymous lifetime.
    ElisionFailure,

    /// This rib forbids usage of generic parameters inside of const parameter types.
    ///
    /// While this is desirable to support eventually, it is difficult to do and so is
    /// currently forbidden. See rust-lang/project-const-generics#28 for more info.
    ConstParamTy,

    /// Usage of generic parameters is forbidden in various positions for anon consts:
    /// - const arguments when `generic_const_exprs` is not enabled
    /// - enum discriminant values
    ///
    /// This rib emits an error when a lifetime would resolve to a lifetime parameter.
    ConcreteAnonConst(NoConstantGenericsReason),

    /// This rib acts as a barrier to forbid reference to lifetimes of a parent item.
    Item,
}

#[derive(Copy, Clone, Debug)]
enum LifetimeBinderKind {
    BareFnType,
    PolyTrait,
    WhereBound,
    Item,
    ConstItem,
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
            Item | ConstItem => "item",
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
    /// Type paths `Path`.
    Type,
    /// Trait paths in bounds or impls.
    Trait(AliasPossibility),
    /// Expression paths `path`, with optional parent context.
    Expr(Option<&'a Expr>),
    /// Paths in path patterns `Path`.
    Pat,
    /// Paths in struct expressions and patterns `Path { .. }`.
    Struct,
    /// Paths in tuple struct patterns `Path(..)`.
    TupleStruct(Span, &'a [Span]),
    /// `m::A::B` in `<T as m::A>::B::C`.
    TraitItem(Namespace),
    /// Paths in delegation item
    Delegation,
    /// An arg in a `use<'a, N>` precise-capturing bound.
    PreciseCapturingArg(Namespace),
    /// Paths that end with `(..)`, for return type notation.
    ReturnTypeNotation,
    /// Paths from `#[define_opaque]` attributes
    DefineOpaques,
}

impl<'a> PathSource<'a> {
    fn namespace(self) -> Namespace {
        match self {
            PathSource::Type
            | PathSource::Trait(_)
            | PathSource::Struct
            | PathSource::DefineOpaques => TypeNS,
            PathSource::Expr(..)
            | PathSource::Pat
            | PathSource::TupleStruct(..)
            | PathSource::Delegation
            | PathSource::ReturnTypeNotation => ValueNS,
            PathSource::TraitItem(ns) => ns,
            PathSource::PreciseCapturingArg(ns) => ns,
        }
    }

    fn defer_to_typeck(self) -> bool {
        match self {
            PathSource::Type
            | PathSource::Expr(..)
            | PathSource::Pat
            | PathSource::Struct
            | PathSource::TupleStruct(..)
            | PathSource::ReturnTypeNotation => true,
            PathSource::Trait(_)
            | PathSource::TraitItem(..)
            | PathSource::DefineOpaques
            | PathSource::Delegation
            | PathSource::PreciseCapturingArg(..) => false,
        }
    }

    fn descr_expected(self) -> &'static str {
        match &self {
            PathSource::DefineOpaques => "type alias or associated type with opaqaue types",
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
                        if let [segment, _] = path.segments.as_slice()
                            && segment.ident.name == kw::PathRoot =>
                    {
                        "external crate"
                    }
                    ExprKind::Path(_, path)
                        if let Some(segment) = path.segments.last()
                            && let Some(c) = segment.ident.to_string().chars().next()
                            && c.is_uppercase() =>
                    {
                        "function, tuple struct or tuple variant"
                    }
                    _ => "function",
                },
                _ => "value",
            },
            PathSource::ReturnTypeNotation | PathSource::Delegation => "function",
            PathSource::PreciseCapturingArg(..) => "type or const parameter",
        }
    }

    fn is_call(self) -> bool {
        matches!(self, PathSource::Expr(Some(&Expr { kind: ExprKind::Call(..), .. })))
    }

    pub(crate) fn is_expected(self, res: Res) -> bool {
        match self {
            PathSource::DefineOpaques => {
                matches!(
                    res,
                    Res::Def(
                        DefKind::Struct
                            | DefKind::Union
                            | DefKind::Enum
                            | DefKind::TyAlias
                            | DefKind::AssocTy,
                        _
                    ) | Res::SelfTyAlias { .. }
                )
            }
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
                        | DefKind::Static { .. }
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
            PathSource::ReturnTypeNotation => match res {
                Res::Def(DefKind::AssocFn, _) => true,
                _ => false,
            },
            PathSource::Delegation => matches!(res, Res::Def(DefKind::Fn | DefKind::AssocFn, _)),
            PathSource::PreciseCapturingArg(ValueNS) => {
                matches!(res, Res::Def(DefKind::ConstParam, _))
            }
            // We allow `SelfTyAlias` here so we can give a more descriptive error later.
            PathSource::PreciseCapturingArg(TypeNS) => matches!(
                res,
                Res::Def(DefKind::TyParam, _) | Res::SelfTyParam { .. } | Res::SelfTyAlias { .. }
            ),
            PathSource::PreciseCapturingArg(MacroNS) => false,
        }
    }

    fn error_code(self, has_unexpected_resolution: bool) -> ErrCode {
        match (self, has_unexpected_resolution) {
            (PathSource::Trait(_), true) => E0404,
            (PathSource::Trait(_), false) => E0405,
            (PathSource::Type | PathSource::DefineOpaques, true) => E0573,
            (PathSource::Type | PathSource::DefineOpaques, false) => E0412,
            (PathSource::Struct, true) => E0574,
            (PathSource::Struct, false) => E0422,
            (PathSource::Expr(..), true) | (PathSource::Delegation, true) => E0423,
            (PathSource::Expr(..), false) | (PathSource::Delegation, false) => E0425,
            (PathSource::Pat | PathSource::TupleStruct(..), true) => E0532,
            (PathSource::Pat | PathSource::TupleStruct(..), false) => E0531,
            (PathSource::TraitItem(..) | PathSource::ReturnTypeNotation, true) => E0575,
            (PathSource::TraitItem(..) | PathSource::ReturnTypeNotation, false) => E0576,
            (PathSource::PreciseCapturingArg(..), true) => E0799,
            (PathSource::PreciseCapturingArg(..), false) => E0800,
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
    NestedUse(&'a Visibility),
}

impl MaybeExported<'_> {
    fn eval(self, r: &Resolver<'_, '_>) -> bool {
        let def_id = match self {
            MaybeExported::Ok(node_id) => Some(r.local_def_id(node_id)),
            MaybeExported::Impl(Some(trait_def_id)) | MaybeExported::ImplItem(Ok(trait_def_id)) => {
                trait_def_id.as_local()
            }
            MaybeExported::Impl(None) => return true,
            MaybeExported::ImplItem(Err(vis)) | MaybeExported::NestedUse(vis) => {
                return vis.kind.is_pub();
            }
        };
        def_id.is_none_or(|def_id| r.effective_visibilities.is_exported(def_id))
    }
}

/// Used for recording UnnecessaryQualification.
#[derive(Debug)]
pub(crate) struct UnnecessaryQualification<'ra> {
    pub binding: LexicalScopeBinding<'ra>,
    pub node_id: NodeId,
    pub path_span: Span,
    pub removal_span: Span,
}

#[derive(Default, Debug)]
struct DiagMetadata<'ast> {
    /// The current trait's associated items' ident, used for diagnostic suggestions.
    current_trait_assoc_items: Option<&'ast [P<AssocItem>]>,

    /// The current self type if inside an impl (used for better errors).
    current_self_type: Option<Ty>,

    /// The current self item if inside an ADT (used for better errors).
    current_self_item: Option<NodeId>,

    /// The current trait (used to suggest).
    current_item: Option<&'ast Item>,

    /// When processing generic arguments and encountering an unresolved ident not found,
    /// suggest introducing a type or const param depending on the context.
    currently_processing_generic_args: bool,

    /// The current enclosing (non-closure) function (used for better errors).
    current_function: Option<(FnKind<'ast>, Span)>,

    /// A list of labels as of yet unused. Labels will be removed from this map when
    /// they are used (in a `break` or `continue` statement)
    unused_labels: FxIndexMap<NodeId, Span>,

    /// Only used for better errors on `let <pat>: <expr, not type>;`.
    current_let_binding: Option<(Span, Option<Span>, Option<Span>)>,

    current_pat: Option<&'ast Pat>,

    /// Used to detect possible `if let` written without `let` and to provide structured suggestion.
    in_if_condition: Option<&'ast Expr>,

    /// Used to detect possible new binding written without `let` and to provide structured suggestion.
    in_assignment: Option<&'ast Expr>,
    is_assign_rhs: bool,

    /// If we are setting an associated type in trait impl, is it a non-GAT type?
    in_non_gat_assoc_type: Option<bool>,

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

struct LateResolutionVisitor<'a, 'ast, 'ra, 'tcx> {
    r: &'a mut Resolver<'ra, 'tcx>,

    /// The module that represents the current item scope.
    parent_scope: ParentScope<'ra>,

    /// The current set of local scopes for types and values.
    ribs: PerNS<Vec<Rib<'ra>>>,

    /// Previous popped `rib`, only used for diagnostic.
    last_block_rib: Option<Rib<'ra>>,

    /// The current set of local scopes, for labels.
    label_ribs: Vec<Rib<'ra, NodeId>>,

    /// The current set of local scopes for lifetimes.
    lifetime_ribs: Vec<LifetimeRib>,

    /// We are looking for lifetimes in an elision context.
    /// The set contains all the resolutions that we encountered so far.
    /// They will be used to determine the correct lifetime for the fn return type.
    /// The `LifetimeElisionCandidate` is used for diagnostics, to suggest introducing named
    /// lifetimes.
    lifetime_elision_candidates: Option<Vec<(LifetimeRes, LifetimeElisionCandidate)>>,

    /// The trait that the current context can refer to.
    current_trait_ref: Option<(Module<'ra>, TraitRef)>,

    /// Fields used to add information to diagnostic errors.
    diag_metadata: Box<DiagMetadata<'ast>>,

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
impl<'ra: 'ast, 'ast, 'tcx> Visitor<'ast> for LateResolutionVisitor<'_, 'ast, 'ra, 'tcx> {
    fn visit_attribute(&mut self, _: &'ast Attribute) {
        // We do not want to resolve expressions that appear in attributes,
        // as they do not correspond to actual code.
    }
    fn visit_item(&mut self, item: &'ast Item) {
        let prev = replace(&mut self.diag_metadata.current_item, Some(item));
        // Always report errors in items we just entered.
        let old_ignore = replace(&mut self.in_func_body, false);
        self.with_lifetime_rib(LifetimeRibKind::Item, |this| this.resolve_item(item));
        self.in_func_body = old_ignore;
        self.diag_metadata.current_item = prev;
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
        bug!("encountered anon const without a manual call to `resolve_anon_const`: {constant:#?}");
    }
    fn visit_expr(&mut self, expr: &'ast Expr) {
        self.resolve_expr(expr, None);
    }
    fn visit_pat(&mut self, p: &'ast Pat) {
        let prev = self.diag_metadata.current_pat;
        self.diag_metadata.current_pat = Some(p);
        visit::walk_pat(self, p);
        self.diag_metadata.current_pat = prev;
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
        let original = replace(&mut self.diag_metadata.current_let_binding, local_spans);
        self.resolve_local(local);
        self.diag_metadata.current_let_binding = original;
    }
    fn visit_ty(&mut self, ty: &'ast Ty) {
        let prev = self.diag_metadata.current_trait_object;
        let prev_ty = self.diag_metadata.current_type_path;
        match &ty.kind {
            TyKind::Ref(None, _) | TyKind::PinnedRef(None, _) => {
                // Elided lifetime in reference: we resolve as if there was some lifetime `'_` with
                // NodeId `ty.id`.
                // This span will be used in case of elision failure.
                let span = self.r.tcx.sess.source_map().start_point(ty.span);
                self.resolve_elided_lifetime(ty.id, span);
                visit::walk_ty(self, ty);
            }
            TyKind::Path(qself, path) => {
                self.diag_metadata.current_type_path = Some(ty);

                // If we have a path that ends with `(..)`, then it must be
                // return type notation. Resolve that path in the *value*
                // namespace.
                let source = if let Some(seg) = path.segments.last()
                    && let Some(args) = &seg.args
                    && matches!(**args, GenericArgs::ParenthesizedElided(..))
                {
                    PathSource::ReturnTypeNotation
                } else {
                    PathSource::Type
                };

                self.smart_resolve_path(ty.id, qself, path, source);

                // Check whether we should interpret this as a bare trait object.
                if qself.is_none()
                    && let Some(partial_res) = self.r.partial_res_map.get(&ty.id)
                    && let Some(Res::Def(DefKind::Trait | DefKind::TraitAlias, _)) =
                        partial_res.full_res()
                {
                    // This path is actually a bare trait object. In case of a bare `Fn`-trait
                    // object with anonymous lifetimes, we need this rib to correctly place the
                    // synthetic lifetimes.
                    let span = ty.span.shrink_to_lo().to(path.span.shrink_to_lo());
                    self.with_generic_param_rib(
                        &[],
                        RibKind::Normal,
                        LifetimeRibKind::Generics {
                            binder: ty.id,
                            kind: LifetimeBinderKind::PolyTrait,
                            span,
                        },
                        |this| this.visit_path(path, ty.id),
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
            TyKind::TraitObject(bounds, ..) => {
                self.diag_metadata.current_trait_object = Some(&bounds[..]);
                visit::walk_ty(self, ty)
            }
            TyKind::BareFn(bare_fn) => {
                let span = ty.span.shrink_to_lo().to(bare_fn.decl_span.shrink_to_lo());
                self.with_generic_param_rib(
                    &bare_fn.generic_params,
                    RibKind::Normal,
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
            TyKind::UnsafeBinder(unsafe_binder) => {
                // FIXME(unsafe_binder): Better span
                let span = ty.span;
                self.with_generic_param_rib(
                    &unsafe_binder.generic_params,
                    RibKind::Normal,
                    LifetimeRibKind::Generics {
                        binder: ty.id,
                        kind: LifetimeBinderKind::BareFnType,
                        span,
                    },
                    |this| {
                        this.visit_generic_params(&unsafe_binder.generic_params, false);
                        this.with_lifetime_rib(
                            // We don't allow anonymous `unsafe &'_ ()` binders,
                            // although I guess we could.
                            LifetimeRibKind::AnonymousReportError,
                            |this| this.visit_ty(&unsafe_binder.inner_ty),
                        );
                    },
                )
            }
            TyKind::Array(element_ty, length) => {
                self.visit_ty(element_ty);
                self.resolve_anon_const(length, AnonConstKind::ConstArg(IsRepeatExpr::No));
            }
            TyKind::Typeof(ct) => {
                self.resolve_anon_const(ct, AnonConstKind::ConstArg(IsRepeatExpr::No))
            }
            _ => visit::walk_ty(self, ty),
        }
        self.diag_metadata.current_trait_object = prev;
        self.diag_metadata.current_type_path = prev_ty;
    }

    fn visit_ty_pat(&mut self, t: &'ast TyPat) -> Self::Result {
        match &t.kind {
            TyPatKind::Range(start, end, _) => {
                if let Some(start) = start {
                    self.resolve_anon_const(start, AnonConstKind::ConstArg(IsRepeatExpr::No));
                }
                if let Some(end) = end {
                    self.resolve_anon_const(end, AnonConstKind::ConstArg(IsRepeatExpr::No));
                }
            }
            TyPatKind::Err(_) => {}
        }
    }

    fn visit_poly_trait_ref(&mut self, tref: &'ast PolyTraitRef) {
        let span = tref.span.shrink_to_lo().to(tref.trait_ref.path.span.shrink_to_lo());
        self.with_generic_param_rib(
            &tref.bound_generic_params,
            RibKind::Normal,
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
        let def_kind = self.r.local_def_kind(foreign_item.id);
        match foreign_item.kind {
            ForeignItemKind::TyAlias(box TyAlias { ref generics, .. }) => {
                self.with_generic_param_rib(
                    &generics.params,
                    RibKind::Item(HasGenericParams::Yes(generics.span), def_kind),
                    LifetimeRibKind::Generics {
                        binder: foreign_item.id,
                        kind: LifetimeBinderKind::Item,
                        span: generics.span,
                    },
                    |this| visit::walk_item(this, foreign_item),
                );
            }
            ForeignItemKind::Fn(box Fn { ref generics, .. }) => {
                self.with_generic_param_rib(
                    &generics.params,
                    RibKind::Item(HasGenericParams::Yes(generics.span), def_kind),
                    LifetimeRibKind::Generics {
                        binder: foreign_item.id,
                        kind: LifetimeBinderKind::Function,
                        span: generics.span,
                    },
                    |this| visit::walk_item(this, foreign_item),
                );
            }
            ForeignItemKind::Static(..) => {
                self.with_static_rib(def_kind, |this| visit::walk_item(this, foreign_item))
            }
            ForeignItemKind::MacCall(..) => {
                panic!("unexpanded macro in resolve!")
            }
        }
    }
    fn visit_fn(&mut self, fn_kind: FnKind<'ast>, sp: Span, fn_id: NodeId) {
        let previous_value = self.diag_metadata.current_function;
        match fn_kind {
            // Bail if the function is foreign, and thus cannot validly have
            // a body, or if there's no body for some other reason.
            FnKind::Fn(FnCtxt::Foreign, _, Fn { sig, ident, generics, .. })
            | FnKind::Fn(_, _, Fn { sig, ident, generics, body: None, .. }) => {
                self.visit_fn_header(&sig.header);
                self.visit_ident(ident);
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
                    },
                );
                return;
            }
            FnKind::Fn(..) => {
                self.diag_metadata.current_function = Some((fn_kind, sp));
            }
            // Do not update `current_function` for closures: it suggests `self` parameters.
            FnKind::Closure(..) => {}
        };
        debug!("(resolving function) entering function");

        // Create a value rib for the function.
        self.with_rib(ValueNS, RibKind::FnOrCoroutine, |this| {
            // Create a label rib for the function.
            this.with_label_rib(RibKind::FnOrCoroutine, |this| {
                match fn_kind {
                    FnKind::Fn(_, _, Fn { sig, generics, contract, body, .. }) => {
                        this.visit_generics(generics);

                        let declaration = &sig.decl;
                        let coro_node_id = sig
                            .header
                            .coroutine_kind
                            .map(|coroutine_kind| coroutine_kind.return_id());

                        this.with_lifetime_rib(
                            LifetimeRibKind::AnonymousCreateParameter {
                                binder: fn_id,
                                report_in_path: coro_node_id.is_some(),
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
                                );
                            },
                        );

                        if let Some(contract) = contract {
                            this.visit_contract(contract);
                        }

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
                    FnKind::Closure(binder, _, declaration, body) => {
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
        self.diag_metadata.current_function = previous_value;
    }

    fn visit_lifetime(&mut self, lifetime: &'ast Lifetime, use_ctxt: visit::LifetimeCtxt) {
        self.resolve_lifetime(lifetime, use_ctxt)
    }

    fn visit_precise_capturing_arg(&mut self, arg: &'ast PreciseCapturingArg) {
        match arg {
            // Lower the lifetime regularly; we'll resolve the lifetime and check
            // it's a parameter later on in HIR lowering.
            PreciseCapturingArg::Lifetime(_) => {}

            PreciseCapturingArg::Arg(path, id) => {
                // we want `impl use<C>` to try to resolve `C` as both a type parameter or
                // a const parameter. Since the resolver specifically doesn't allow having
                // two generic params with the same name, even if they're a different namespace,
                // it doesn't really matter which we try resolving first, but just like
                // `Ty::Param` we just fall back to the value namespace only if it's missing
                // from the type namespace.
                let mut check_ns = |ns| {
                    self.maybe_resolve_ident_in_lexical_scope(path.segments[0].ident, ns).is_some()
                };
                // Like `Ty::Param`, we try resolving this as both a const and a type.
                if !check_ns(TypeNS) && check_ns(ValueNS) {
                    self.smart_resolve_path(
                        *id,
                        &None,
                        path,
                        PathSource::PreciseCapturingArg(ValueNS),
                    );
                } else {
                    self.smart_resolve_path(
                        *id,
                        &None,
                        path,
                        PathSource::PreciseCapturingArg(TypeNS),
                    );
                }
            }
        }

        visit::walk_precise_capturing_arg(self, arg)
    }

    fn visit_generics(&mut self, generics: &'ast Generics) {
        self.visit_generic_params(&generics.params, self.diag_metadata.current_self_item.is_some());
        for p in &generics.where_clause.predicates {
            self.visit_where_predicate(p);
        }
    }

    fn visit_closure_binder(&mut self, b: &'ast ClosureBinder) {
        match b {
            ClosureBinder::NotPresent => {}
            ClosureBinder::For { generic_params, .. } => {
                self.visit_generic_params(
                    generic_params,
                    self.diag_metadata.current_self_item.is_some(),
                );
            }
        }
    }

    fn visit_generic_arg(&mut self, arg: &'ast GenericArg) {
        debug!("visit_generic_arg({:?})", arg);
        let prev = replace(&mut self.diag_metadata.currently_processing_generic_args, true);
        match arg {
            GenericArg::Type(ty) => {
                // We parse const arguments as path types as we cannot distinguish them during
                // parsing. We try to resolve that ambiguity by attempting resolution the type
                // namespace first, and if that fails we try again in the value namespace. If
                // resolution in the value namespace succeeds, we have an generic const argument on
                // our hands.
                if let TyKind::Path(None, ref path) = ty.kind
                    // We cannot disambiguate multi-segment paths right now as that requires type
                    // checking.
                    && path.is_potential_trivial_const_arg(false)
                {
                    let mut check_ns = |ns| {
                        self.maybe_resolve_ident_in_lexical_scope(path.segments[0].ident, ns)
                            .is_some()
                    };
                    if !check_ns(TypeNS) && check_ns(ValueNS) {
                        self.resolve_anon_const_manual(
                            true,
                            AnonConstKind::ConstArg(IsRepeatExpr::No),
                            |this| {
                                this.smart_resolve_path(ty.id, &None, path, PathSource::Expr(None));
                                this.visit_path(path, ty.id);
                            },
                        );

                        self.diag_metadata.currently_processing_generic_args = prev;
                        return;
                    }
                }

                self.visit_ty(ty);
            }
            GenericArg::Lifetime(lt) => self.visit_lifetime(lt, visit::LifetimeCtxt::GenericArg),
            GenericArg::Const(ct) => {
                self.resolve_anon_const(ct, AnonConstKind::ConstArg(IsRepeatExpr::No))
            }
        }
        self.diag_metadata.currently_processing_generic_args = prev;
    }

    fn visit_assoc_item_constraint(&mut self, constraint: &'ast AssocItemConstraint) {
        self.visit_ident(&constraint.ident);
        if let Some(ref gen_args) = constraint.gen_args {
            // Forbid anonymous lifetimes in GAT parameters until proper semantics are decided.
            self.with_lifetime_rib(LifetimeRibKind::AnonymousReportError, |this| {
                this.visit_generic_args(gen_args)
            });
        }
        match constraint.kind {
            AssocItemConstraintKind::Equality { ref term } => match term {
                Term::Ty(ty) => self.visit_ty(ty),
                Term::Const(c) => {
                    self.resolve_anon_const(c, AnonConstKind::ConstArg(IsRepeatExpr::No))
                }
            },
            AssocItemConstraintKind::Bound { ref bounds } => {
                walk_list!(self, visit_param_bound, bounds, BoundKind::Bound);
            }
        }
    }

    fn visit_path_segment(&mut self, path_segment: &'ast PathSegment) {
        let Some(ref args) = path_segment.args else {
            return;
        };

        match &**args {
            GenericArgs::AngleBracketed(..) => visit::walk_generic_args(self, args),
            GenericArgs::Parenthesized(p_args) => {
                // Probe the lifetime ribs to know how to behave.
                for rib in self.lifetime_ribs.iter().rev() {
                    match rib.kind {
                        // We are inside a `PolyTraitRef`. The lifetimes are
                        // to be introduced in that (maybe implicit) `for<>` binder.
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
                        | LifetimeRibKind::StaticIfNoLifetimeInScope { .. }
                        | LifetimeRibKind::Elided(_)
                        | LifetimeRibKind::ElisionFailure
                        | LifetimeRibKind::ConcreteAnonConst(_)
                        | LifetimeRibKind::ConstParamTy => {}
                    }
                }
            }
            GenericArgs::ParenthesizedElided(_) => {}
        }
    }

    fn visit_where_predicate(&mut self, p: &'ast WherePredicate) {
        debug!("visit_where_predicate {:?}", p);
        let previous_value = replace(&mut self.diag_metadata.current_where_predicate, Some(p));
        self.with_lifetime_rib(LifetimeRibKind::AnonymousReportError, |this| {
            if let WherePredicateKind::BoundPredicate(WhereBoundPredicate {
                bounded_ty,
                bounds,
                bound_generic_params,
                ..
            }) = &p.kind
            {
                let span = p.span.shrink_to_lo().to(bounded_ty.span.shrink_to_lo());
                this.with_generic_param_rib(
                    bound_generic_params,
                    RibKind::Normal,
                    LifetimeRibKind::Generics {
                        binder: bounded_ty.id,
                        kind: LifetimeBinderKind::WhereBound,
                        span,
                    },
                    |this| {
                        this.visit_generic_params(bound_generic_params, false);
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
        self.diag_metadata.current_where_predicate = previous_value;
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
                    self.resolve_anon_const(anon_const, AnonConstKind::InlineConst);
                }
                InlineAsmOperand::Sym { sym } => self.visit_inline_asm_sym(sym),
                InlineAsmOperand::Label { block } => self.visit_block(block),
            }
        }
    }

    fn visit_inline_asm_sym(&mut self, sym: &'ast InlineAsmSym) {
        // This is similar to the code for AnonConst.
        self.with_rib(ValueNS, RibKind::InlineAsmSym, |this| {
            this.with_rib(TypeNS, RibKind::InlineAsmSym, |this| {
                this.with_label_rib(RibKind::InlineAsmSym, |this| {
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

    fn visit_variant_discr(&mut self, discr: &'ast AnonConst) {
        self.resolve_anon_const(discr, AnonConstKind::EnumDiscriminant);
    }

    fn visit_field_def(&mut self, f: &'ast FieldDef) {
        self.resolve_doc_links(&f.attrs, MaybeExported::Ok(f.id));
        let FieldDef {
            attrs,
            id: _,
            span: _,
            vis,
            ident,
            ty,
            is_placeholder: _,
            default,
            safety: _,
        } = f;
        walk_list!(self, visit_attribute, attrs);
        try_visit!(self.visit_vis(vis));
        visit_opt!(self, visit_ident, ident);
        try_visit!(self.visit_ty(ty));
        if let Some(v) = &default {
            self.resolve_anon_const(v, AnonConstKind::FieldDefaultValue);
        }
    }
}

impl<'a, 'ast, 'ra: 'ast, 'tcx> LateResolutionVisitor<'a, 'ast, 'ra, 'tcx> {
    fn new(resolver: &'a mut Resolver<'ra, 'tcx>) -> LateResolutionVisitor<'a, 'ast, 'ra, 'tcx> {
        // During late resolution we only track the module component of the parent scope,
        // although it may be useful to track other components as well for diagnostics.
        let graph_root = resolver.graph_root;
        let parent_scope = ParentScope::module(graph_root, resolver);
        let start_rib_kind = RibKind::Module(graph_root);
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
            diag_metadata: Default::default(),
            // errors at module scope should always be reported
            in_func_body: false,
            lifetime_uses: Default::default(),
        }
    }

    fn maybe_resolve_ident_in_lexical_scope(
        &mut self,
        ident: Ident,
        ns: Namespace,
    ) -> Option<LexicalScopeBinding<'ra>> {
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
        ignore_binding: Option<NameBinding<'ra>>,
    ) -> Option<LexicalScopeBinding<'ra>> {
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
    ) -> PathResult<'ra> {
        self.r.resolve_path_with_ribs(
            path,
            opt_ns,
            &self.parent_scope,
            finalize,
            Some(&self.ribs),
            None,
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
        kind: RibKind<'ra>,
        work: impl FnOnce(&mut Self) -> T,
    ) -> T {
        self.ribs[ns].push(Rib::new(kind));
        let ret = work(self);
        self.ribs[ns].pop();
        ret
    }

    fn with_mod_rib<T>(&mut self, id: NodeId, f: impl FnOnce(&mut Self) -> T) -> T {
        let module = self.r.expect_module(self.r.local_def_id(id).to_def_id());
        // Move down in the graph.
        let orig_module = replace(&mut self.parent_scope.module, module);
        self.with_rib(ValueNS, RibKind::Module(module), |this| {
            this.with_rib(TypeNS, RibKind::Module(module), |this| {
                let ret = f(this);
                this.parent_scope.module = orig_module;
                ret
            })
        })
    }

    fn visit_generic_params(&mut self, params: &'ast [GenericParam], add_self_upper: bool) {
        // For type parameter defaults, we have to ban access
        // to following type parameters, as the GenericArgs can only
        // provide previous type parameters as they're built. We
        // put all the parameters on the ban list and then remove
        // them one by one as they are processed and become available.
        let mut forward_ty_ban_rib =
            Rib::new(RibKind::ForwardGenericParamBan(ForwardGenericParamBanReason::Default));
        let mut forward_const_ban_rib =
            Rib::new(RibKind::ForwardGenericParamBan(ForwardGenericParamBanReason::Default));
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

        // NOTE: We use different ribs here not for a technical reason, but just
        // for better diagnostics.
        let mut forward_ty_ban_rib_const_param_ty = Rib {
            bindings: forward_ty_ban_rib.bindings.clone(),
            patterns_with_skipped_bindings: Default::default(),
            kind: RibKind::ForwardGenericParamBan(ForwardGenericParamBanReason::ConstParamTy),
        };
        let mut forward_const_ban_rib_const_param_ty = Rib {
            bindings: forward_const_ban_rib.bindings.clone(),
            patterns_with_skipped_bindings: Default::default(),
            kind: RibKind::ForwardGenericParamBan(ForwardGenericParamBanReason::ConstParamTy),
        };
        // We'll ban these with a `ConstParamTy` rib, so just clear these ribs for better
        // diagnostics, so we don't mention anything about const param tys having generics at all.
        if !self.r.tcx.features().generic_const_parameter_types() {
            forward_ty_ban_rib_const_param_ty.bindings.clear();
            forward_const_ban_rib_const_param_ty.bindings.clear();
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

                        if let Some(ty) = default {
                            this.ribs[TypeNS].push(forward_ty_ban_rib);
                            this.ribs[ValueNS].push(forward_const_ban_rib);
                            this.visit_ty(ty);
                            forward_const_ban_rib = this.ribs[ValueNS].pop().unwrap();
                            forward_ty_ban_rib = this.ribs[TypeNS].pop().unwrap();
                        }

                        // Allow all following defaults to refer to this type parameter.
                        let i = &Ident::with_dummy_span(param.ident.name);
                        forward_ty_ban_rib.bindings.swap_remove(i);
                        forward_ty_ban_rib_const_param_ty.bindings.swap_remove(i);
                    }
                    GenericParamKind::Const { ref ty, kw_span: _, ref default } => {
                        // Const parameters can't have param bounds.
                        assert!(param.bounds.is_empty());

                        this.ribs[TypeNS].push(forward_ty_ban_rib_const_param_ty);
                        this.ribs[ValueNS].push(forward_const_ban_rib_const_param_ty);
                        if this.r.tcx.features().generic_const_parameter_types() {
                            this.visit_ty(ty)
                        } else {
                            this.ribs[TypeNS].push(Rib::new(RibKind::ConstParamTy));
                            this.ribs[ValueNS].push(Rib::new(RibKind::ConstParamTy));
                            this.with_lifetime_rib(LifetimeRibKind::ConstParamTy, |this| {
                                this.visit_ty(ty)
                            });
                            this.ribs[TypeNS].pop().unwrap();
                            this.ribs[ValueNS].pop().unwrap();
                        }
                        forward_const_ban_rib_const_param_ty = this.ribs[ValueNS].pop().unwrap();
                        forward_ty_ban_rib_const_param_ty = this.ribs[TypeNS].pop().unwrap();

                        if let Some(expr) = default {
                            this.ribs[TypeNS].push(forward_ty_ban_rib);
                            this.ribs[ValueNS].push(forward_const_ban_rib);
                            this.resolve_anon_const(
                                expr,
                                AnonConstKind::ConstArg(IsRepeatExpr::No),
                            );
                            forward_const_ban_rib = this.ribs[ValueNS].pop().unwrap();
                            forward_ty_ban_rib = this.ribs[TypeNS].pop().unwrap();
                        }

                        // Allow all following defaults to refer to this const parameter.
                        let i = &Ident::with_dummy_span(param.ident.name);
                        forward_const_ban_rib.bindings.swap_remove(i);
                        forward_const_ban_rib_const_param_ty.bindings.swap_remove(i);
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
                LifetimeRes::Static { suppress_elision_warning: false },
                LifetimeElisionCandidate::Named,
            );
            return;
        }

        if ident.name == kw::UnderscoreLifetime {
            return self.resolve_anonymous_lifetime(lifetime, lifetime.id, false);
        }

        let mut lifetime_rib_iter = self.lifetime_ribs.iter().rev();
        while let Some(rib) = lifetime_rib_iter.next() {
            let normalized_ident = ident.normalize_to_macros_2_0();
            if let Some(&(_, res)) = rib.bindings.get(&normalized_ident) {
                self.record_lifetime_res(lifetime.id, res, LifetimeElisionCandidate::Named);

                if let LifetimeRes::Param { param, binder } = res {
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
                                    | LifetimeRibKind::StaticIfNoLifetimeInScope { .. }
                                    | LifetimeRibKind::ElisionFailure => Some(LifetimeUseSet::Many),
                                    // An anonymous lifetime is legal here, and bound to the right
                                    // place, go ahead.
                                    LifetimeRibKind::AnonymousCreateParameter {
                                        binder: anon_binder,
                                        ..
                                    } => Some(if binder == anon_binder {
                                        LifetimeUseSet::One { use_span: ident.span, use_ctxt }
                                    } else {
                                        LifetimeUseSet::Many
                                    }),
                                    // Only report if eliding the lifetime would have the same
                                    // semantics.
                                    LifetimeRibKind::Elided(r) => Some(if res == r {
                                        LifetimeUseSet::One { use_span: ident.span, use_ctxt }
                                    } else {
                                        LifetimeUseSet::Many
                                    }),
                                    LifetimeRibKind::Generics { .. }
                                    | LifetimeRibKind::ConstParamTy => None,
                                    LifetimeRibKind::ConcreteAnonConst(_) => {
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
                LifetimeRibKind::ConstParamTy => {
                    self.emit_non_static_lt_in_const_param_ty_error(lifetime);
                    self.record_lifetime_res(
                        lifetime.id,
                        LifetimeRes::Error,
                        LifetimeElisionCandidate::Ignore,
                    );
                    return;
                }
                LifetimeRibKind::ConcreteAnonConst(cause) => {
                    self.emit_forbidden_non_static_lifetime_error(cause, lifetime);
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
                | LifetimeRibKind::AnonymousReportError
                | LifetimeRibKind::StaticIfNoLifetimeInScope { .. } => {}
            }
        }

        let normalized_ident = ident.normalize_to_macros_2_0();
        let outer_res = lifetime_rib_iter
            .find_map(|rib| rib.bindings.get_key_value(&normalized_ident).map(|(&outer, _)| outer));

        self.emit_undeclared_lifetime_error(lifetime, outer_res);
        self.record_lifetime_res(lifetime.id, LifetimeRes::Error, LifetimeElisionCandidate::Named);
    }

    #[instrument(level = "debug", skip(self))]
    fn resolve_anonymous_lifetime(
        &mut self,
        lifetime: &Lifetime,
        id_for_lint: NodeId,
        elided: bool,
    ) {
        debug_assert_eq!(lifetime.ident.name, kw::UnderscoreLifetime);

        let kind =
            if elided { MissingLifetimeKind::Ampersand } else { MissingLifetimeKind::Underscore };
        let missing_lifetime = MissingLifetime {
            id: lifetime.id,
            span: lifetime.ident.span,
            kind,
            count: 1,
            id_for_lint,
        };
        let elision_candidate = LifetimeElisionCandidate::Missing(missing_lifetime);
        for (i, rib) in self.lifetime_ribs.iter().enumerate().rev() {
            debug!(?rib.kind);
            match rib.kind {
                LifetimeRibKind::AnonymousCreateParameter { binder, .. } => {
                    let res = self.create_fresh_lifetime(lifetime.ident, binder, kind);
                    self.record_lifetime_res(lifetime.id, res, elision_candidate);
                    return;
                }
                LifetimeRibKind::StaticIfNoLifetimeInScope { lint_id: node_id, emit_lint } => {
                    let mut lifetimes_in_scope = vec![];
                    for rib in self.lifetime_ribs[..i].iter().rev() {
                        lifetimes_in_scope.extend(rib.bindings.iter().map(|(ident, _)| ident.span));
                        // Consider any anonymous lifetimes, too
                        if let LifetimeRibKind::AnonymousCreateParameter { binder, .. } = rib.kind
                            && let Some(extra) = self.r.extra_lifetime_params_map.get(&binder)
                        {
                            lifetimes_in_scope.extend(extra.iter().map(|(ident, _, _)| ident.span));
                        }
                        if let LifetimeRibKind::Item = rib.kind {
                            break;
                        }
                    }
                    if lifetimes_in_scope.is_empty() {
                        self.record_lifetime_res(
                            lifetime.id,
                            // We are inside a const item, so do not warn.
                            LifetimeRes::Static { suppress_elision_warning: true },
                            elision_candidate,
                        );
                        return;
                    } else if emit_lint {
                        self.r.lint_buffer.buffer_lint(
                            lint::builtin::ELIDED_LIFETIMES_IN_ASSOCIATED_CONSTANT,
                            node_id,
                            lifetime.ident.span,
                            lint::BuiltinLintDiag::AssociatedConstElidedLifetime {
                                elided,
                                span: lifetime.ident.span,
                                lifetimes_in_scope: lifetimes_in_scope.into(),
                            },
                        );
                    }
                }
                LifetimeRibKind::AnonymousReportError => {
                    if elided {
                        let suggestion = self.lifetime_ribs[i..].iter().rev().find_map(|rib| {
                            if let LifetimeRibKind::Generics {
                                span,
                                kind: LifetimeBinderKind::PolyTrait | LifetimeBinderKind::WhereBound,
                                ..
                            } = rib.kind
                            {
                                Some(errors::ElidedAnonymousLivetimeReportErrorSuggestion {
                                    lo: span.shrink_to_lo(),
                                    hi: lifetime.ident.span.shrink_to_hi(),
                                })
                            } else {
                                None
                            }
                        });
                        // are we trying to use an anonymous lifetime
                        // on a non GAT associated trait type?
                        if !self.in_func_body
                            && let Some((module, _)) = &self.current_trait_ref
                            && let Some(ty) = &self.diag_metadata.current_self_type
                            && Some(true) == self.diag_metadata.in_non_gat_assoc_type
                            && let crate::ModuleKind::Def(DefKind::Trait, trait_id, _) = module.kind
                        {
                            if def_id_matches_path(
                                self.r.tcx,
                                trait_id,
                                &["core", "iter", "traits", "iterator", "Iterator"],
                            ) {
                                self.r.dcx().emit_err(errors::LendingIteratorReportError {
                                    lifetime: lifetime.ident.span,
                                    ty: ty.span,
                                });
                            } else {
                                self.r.dcx().emit_err(errors::AnonymousLivetimeNonGatReportError {
                                    lifetime: lifetime.ident.span,
                                });
                            }
                        } else {
                            self.r.dcx().emit_err(errors::ElidedAnonymousLivetimeReportError {
                                span: lifetime.ident.span,
                                suggestion,
                            });
                        }
                    } else {
                        self.r.dcx().emit_err(errors::ExplicitAnonymousLivetimeReportError {
                            span: lifetime.ident.span,
                        });
                    };
                    self.record_lifetime_res(lifetime.id, LifetimeRes::Error, elision_candidate);
                    return;
                }
                LifetimeRibKind::Elided(res) => {
                    self.record_lifetime_res(lifetime.id, res, elision_candidate);
                    return;
                }
                LifetimeRibKind::ElisionFailure => {
                    self.diag_metadata.current_elision_failures.push(missing_lifetime);
                    self.record_lifetime_res(lifetime.id, LifetimeRes::Error, elision_candidate);
                    return;
                }
                LifetimeRibKind::Item => break,
                LifetimeRibKind::Generics { .. } | LifetimeRibKind::ConstParamTy => {}
                LifetimeRibKind::ConcreteAnonConst(_) => {
                    // There is always an `Elided(LifetimeRes::Infer)` inside an `AnonConst`.
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
        self.resolve_anonymous_lifetime(&lt, anchor_id, true);
    }

    #[instrument(level = "debug", skip(self))]
    fn create_fresh_lifetime(
        &mut self,
        ident: Ident,
        binder: NodeId,
        kind: MissingLifetimeKind,
    ) -> LifetimeRes {
        debug_assert_eq!(ident.name, kw::UnderscoreLifetime);
        debug!(?ident.span);

        // Leave the responsibility to create the `LocalDefId` to lowering.
        let param = self.r.next_node_id();
        let res = LifetimeRes::Fresh { param, binder, kind };
        self.record_lifetime_param(param, res);

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
                PathSource::Trait(..)
                | PathSource::TraitItem(..)
                | PathSource::Type
                | PathSource::PreciseCapturingArg(..)
                | PathSource::ReturnTypeNotation => false,
                PathSource::Expr(..)
                | PathSource::Pat
                | PathSource::Struct
                | PathSource::TupleStruct(..)
                | PathSource::DefineOpaques
                | PathSource::Delegation => true,
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

            let kind = if segment.has_generic_args {
                MissingLifetimeKind::Comma
            } else {
                MissingLifetimeKind::Brackets
            };
            let missing_lifetime = MissingLifetime {
                id: node_ids.start,
                id_for_lint: segment_id,
                span: elided_lifetime_span,
                kind,
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
                    LifetimeRibKind::AnonymousCreateParameter { report_in_path: true, .. }
                    | LifetimeRibKind::StaticIfNoLifetimeInScope { .. } => {
                        let sess = self.r.tcx.sess;
                        let subdiag = rustc_errors::elided_lifetime_in_path_suggestion(
                            sess.source_map(),
                            expected_lifetimes,
                            path_span,
                            !segment.has_generic_args,
                            elided_lifetime_span,
                        );
                        self.r.dcx().emit_err(errors::ImplicitElidedLifetimeNotAllowedHere {
                            span: path_span,
                            subdiag,
                        });
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
                            let res = self.create_fresh_lifetime(ident, binder, kind);
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
                        self.diag_metadata.current_elision_failures.push(missing_lifetime);
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
                    LifetimeRibKind::Generics { .. } | LifetimeRibKind::ConstParamTy => {}
                    LifetimeRibKind::ConcreteAnonConst(_) => {
                        // There is always an `Elided(LifetimeRes::Infer)` inside an `AnonConst`.
                        span_bug!(elided_lifetime_span, "unexpected rib kind: {:?}", rib.kind)
                    }
                }
            }

            if should_lint {
                self.r.lint_buffer.buffer_lint(
                    lint::builtin::ELIDED_LIFETIMES_IN_PATHS,
                    segment_id,
                    elided_lifetime_span,
                    lint::BuiltinLintDiag::ElidedLifetimesInPaths(
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
            panic!("lifetime {id:?} resolved multiple times ({prev_res:?} before, {res:?} now)")
        }

        match candidate {
            LifetimeElisionCandidate::Missing(missing @ MissingLifetime { .. }) => {
                debug_assert_eq!(id, missing.id);
                match res {
                    LifetimeRes::Static { suppress_elision_warning } => {
                        if !suppress_elision_warning {
                            self.r.lint_buffer.buffer_lint(
                                lint::builtin::ELIDED_NAMED_LIFETIMES,
                                missing.id_for_lint,
                                missing.span,
                                BuiltinLintDiag::ElidedNamedLifetimes {
                                    elided: (missing.span, missing.kind),
                                    resolution: lint::ElidedLifetimeResolution::Static,
                                },
                            );
                        }
                    }
                    LifetimeRes::Param { param, binder: _ } => {
                        let tcx = self.r.tcx();
                        self.r.lint_buffer.buffer_lint(
                            lint::builtin::ELIDED_NAMED_LIFETIMES,
                            missing.id_for_lint,
                            missing.span,
                            BuiltinLintDiag::ElidedNamedLifetimes {
                                elided: (missing.span, missing.kind),
                                resolution: lint::ElidedLifetimeResolution::Param(
                                    tcx.item_name(param.into()),
                                    tcx.source_span(param),
                                ),
                            },
                        );
                    }
                    LifetimeRes::Fresh { .. }
                    | LifetimeRes::Infer
                    | LifetimeRes::Error
                    | LifetimeRes::ElidedAnchor { .. } => {}
                }
            }
            LifetimeElisionCandidate::Ignore | LifetimeElisionCandidate::Named => {}
        }

        match res {
            LifetimeRes::Param { .. } | LifetimeRes::Fresh { .. } | LifetimeRes::Static { .. } => {
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
                "lifetime parameter {id:?} resolved multiple times ({prev_res:?} before, {res:?} now)"
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

        let outer_failures = take(&mut self.diag_metadata.current_elision_failures);
        let output_rib = if let Ok(res) = elision_lifetime.as_ref() {
            self.r.lifetime_elision_allowed.insert(fn_id);
            LifetimeRibKind::Elided(*res)
        } else {
            LifetimeRibKind::ElisionFailure
        };
        self.with_lifetime_rib(output_rib, |this| visit::walk_fn_ret_ty(this, output_ty));
        let elision_failures =
            replace(&mut self.diag_metadata.current_elision_failures, outer_failures);
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
                let distinct: UnordSet<_> = candidates.iter().map(|(res, _)| *res).collect();
                let lifetime_count = distinct.len();
                if lifetime_count != 0 {
                    parameter_info.push(ElisionFnParameter {
                        index,
                        ident: if let Some(pat) = pat
                            && let PatKind::Ident(_, ident, _) = pat.kind
                        {
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
                if !distinct.is_empty() {
                    match elision_lifetime {
                        // We are the first parameter to bind lifetimes.
                        Elision::None => {
                            if let Some(res) = distinct.get_only() {
                                // We have a single lifetime => success.
                                elision_lifetime = Elision::Param(*res)
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
                elision_lifetime = match self_lifetime {
                    // We found `self` elision.
                    Set1::One(lifetime) => Elision::Self_(lifetime),
                    // `self` itself had ambiguous lifetimes, e.g.
                    // &Box<&Self>. In this case we won't consider
                    // taking an alternative parameter lifetime; just avoid elision
                    // entirely.
                    Set1::Many => Elision::Err,
                    // We do not have `self` elision: disregard the `Elision::Param` that we may
                    // have found.
                    Set1::Empty => Elision::None,
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
        /// Visits a type to find all the &references, and determines the
        /// set of lifetimes for all of those references where the referent
        /// contains Self.
        struct FindReferenceVisitor<'a, 'ra, 'tcx> {
            r: &'a Resolver<'ra, 'tcx>,
            impl_self: Option<Res>,
            lifetime: Set1<LifetimeRes>,
        }

        impl<'ra> Visitor<'ra> for FindReferenceVisitor<'_, '_, '_> {
            fn visit_ty(&mut self, ty: &'ra Ty) {
                trace!("FindReferenceVisitor considering ty={:?}", ty);
                if let TyKind::Ref(lt, _) | TyKind::PinnedRef(lt, _) = ty.kind {
                    // See if anything inside the &thing contains Self
                    let mut visitor =
                        SelfVisitor { r: self.r, impl_self: self.impl_self, self_found: false };
                    visitor.visit_ty(ty);
                    trace!("FindReferenceVisitor: SelfVisitor self_found={:?}", visitor.self_found);
                    if visitor.self_found {
                        let lt_id = if let Some(lt) = lt {
                            lt.id
                        } else {
                            let res = self.r.lifetimes_res_map[&ty.id];
                            let LifetimeRes::ElidedAnchor { start, .. } = res else { bug!() };
                            start
                        };
                        let lt_res = self.r.lifetimes_res_map[&lt_id];
                        trace!("FindReferenceVisitor inserting res={:?}", lt_res);
                        self.lifetime.insert(lt_res);
                    }
                }
                visit::walk_ty(self, ty)
            }

            // A type may have an expression as a const generic argument.
            // We do not want to recurse into those.
            fn visit_expr(&mut self, _: &'ra Expr) {}
        }

        /// Visitor which checks the referent of a &Thing to see if the
        /// Thing contains Self
        struct SelfVisitor<'a, 'ra, 'tcx> {
            r: &'a Resolver<'ra, 'tcx>,
            impl_self: Option<Res>,
            self_found: bool,
        }

        impl SelfVisitor<'_, '_, '_> {
            // Look for `self: &'a Self` - also desugared from `&'a self`
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

        impl<'ra> Visitor<'ra> for SelfVisitor<'_, '_, '_> {
            fn visit_ty(&mut self, ty: &'ra Ty) {
                trace!("SelfVisitor considering ty={:?}", ty);
                if self.is_self_ty(ty) {
                    trace!("SelfVisitor found Self");
                    self.self_found = true;
                }
                visit::walk_ty(self, ty)
            }

            // A type may have an expression as a const generic argument.
            // We do not want to recurse into those.
            fn visit_expr(&mut self, _: &'ra Expr) {}
        }

        let impl_self = self
            .diag_metadata
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
        let mut visitor = FindReferenceVisitor { r: self.r, impl_self, lifetime: Set1::Empty };
        visitor.visit_ty(ty);
        trace!("FindReferenceVisitor found={:?}", visitor.lifetime);
        visitor.lifetime
    }

    /// Searches the current set of local scopes for labels. Returns the `NodeId` of the resolved
    /// label and reports an error if the label is not found or is unreachable.
    fn resolve_label(&mut self, mut label: Ident) -> Result<(NodeId, Span), ResolutionError<'ra>> {
        let mut suggestion = None;

        for i in (0..self.label_ribs.len()).rev() {
            let rib = &self.label_ribs[i];

            if let RibKind::MacroDefinition(def) = rib.kind
                // If an invocation of this macro created `ident`, give up on `ident`
                // and switch to `ident`'s source from the macro definition.
                && def == self.r.macro_def(label.span.ctxt())
            {
                label.span.remove_mark();
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
        ribs.iter().all(|rib| !rib.kind.is_label_barrier())
    }

    fn resolve_adt(&mut self, item: &'ast Item, generics: &'ast Generics) {
        debug!("resolve_adt");
        let kind = self.r.local_def_kind(item.id);
        self.with_current_self_item(item, |this| {
            this.with_generic_param_rib(
                &generics.params,
                RibKind::Item(HasGenericParams::Yes(generics.span), kind),
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
        if let [segment, rest @ ..] = use_tree.prefix.segments.as_slice() {
            let ident = segment.ident;
            if ident.is_path_segment_keyword() || ident.span.is_rust_2015() {
                return;
            }

            let nss = match use_tree.kind {
                UseTreeKind::Simple(..) if rest.is_empty() => &[TypeNS, ValueNS][..],
                _ => &[TypeNS],
            };
            let report_error = |this: &Self, ns| {
                if this.should_report_errs() {
                    let what = if ns == TypeNS { "type parameters" } else { "local variables" };
                    this.r.dcx().emit_err(errors::ImportsCannotReferTo { span: ident.span, what });
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
        } else if let UseTreeKind::Nested { items, .. } = &use_tree.kind {
            for (use_tree, _) in items {
                self.future_proof_import(use_tree);
            }
        }
    }

    fn resolve_item(&mut self, item: &'ast Item) {
        let mod_inner_docs =
            matches!(item.kind, ItemKind::Mod(..)) && rustdoc::inner_docs(&item.attrs);
        if !mod_inner_docs && !matches!(item.kind, ItemKind::Impl(..) | ItemKind::Use(..)) {
            self.resolve_doc_links(&item.attrs, MaybeExported::Ok(item.id));
        }

        debug!("(resolving item) resolving {:?} ({:?})", item.kind.ident(), item.kind);

        let def_kind = self.r.local_def_kind(item.id);
        match item.kind {
            ItemKind::TyAlias(box TyAlias { ref generics, .. }) => {
                self.with_generic_param_rib(
                    &generics.params,
                    RibKind::Item(HasGenericParams::Yes(generics.span), def_kind),
                    LifetimeRibKind::Generics {
                        binder: item.id,
                        kind: LifetimeBinderKind::Item,
                        span: generics.span,
                    },
                    |this| visit::walk_item(this, item),
                );
            }

            ItemKind::Fn(box Fn { ref generics, ref define_opaque, .. }) => {
                self.with_generic_param_rib(
                    &generics.params,
                    RibKind::Item(HasGenericParams::Yes(generics.span), def_kind),
                    LifetimeRibKind::Generics {
                        binder: item.id,
                        kind: LifetimeBinderKind::Function,
                        span: generics.span,
                    },
                    |this| visit::walk_item(this, item),
                );
                self.resolve_define_opaques(define_opaque);
            }

            ItemKind::Enum(_, _, ref generics)
            | ItemKind::Struct(_, _, ref generics)
            | ItemKind::Union(_, _, ref generics) => {
                self.resolve_adt(item, generics);
            }

            ItemKind::Impl(box Impl {
                ref generics,
                ref of_trait,
                ref self_ty,
                items: ref impl_items,
                ..
            }) => {
                self.diag_metadata.current_impl_items = Some(impl_items);
                self.resolve_implementation(
                    &item.attrs,
                    generics,
                    of_trait,
                    self_ty,
                    item.id,
                    impl_items,
                );
                self.diag_metadata.current_impl_items = None;
            }

            ItemKind::Trait(box Trait { ref generics, ref bounds, ref items, .. }) => {
                // Create a new rib for the trait-wide type parameters.
                self.with_generic_param_rib(
                    &generics.params,
                    RibKind::Item(HasGenericParams::Yes(generics.span), def_kind),
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

            ItemKind::TraitAlias(_, ref generics, ref bounds) => {
                // Create a new rib for the trait-wide type parameters.
                self.with_generic_param_rib(
                    &generics.params,
                    RibKind::Item(HasGenericParams::Yes(generics.span), def_kind),
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
                self.with_mod_rib(item.id, |this| {
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

            ItemKind::Static(box ast::StaticItem {
                ident,
                ref ty,
                ref expr,
                ref define_opaque,
                ..
            }) => {
                self.with_static_rib(def_kind, |this| {
                    this.with_lifetime_rib(
                        LifetimeRibKind::Elided(LifetimeRes::Static {
                            suppress_elision_warning: true,
                        }),
                        |this| {
                            this.visit_ty(ty);
                        },
                    );
                    if let Some(expr) = expr {
                        // We already forbid generic params because of the above item rib,
                        // so it doesn't matter whether this is a trivial constant.
                        this.resolve_const_body(expr, Some((ident, ConstantItemKind::Static)));
                    }
                });
                self.resolve_define_opaques(define_opaque);
            }

            ItemKind::Const(box ast::ConstItem {
                ident,
                ref generics,
                ref ty,
                ref expr,
                ref define_opaque,
                ..
            }) => {
                self.with_generic_param_rib(
                    &generics.params,
                    RibKind::Item(
                        if self.r.tcx.features().generic_const_items() {
                            HasGenericParams::Yes(generics.span)
                        } else {
                            HasGenericParams::No
                        },
                        def_kind,
                    ),
                    LifetimeRibKind::Generics {
                        binder: item.id,
                        kind: LifetimeBinderKind::ConstItem,
                        span: generics.span,
                    },
                    |this| {
                        this.visit_generics(generics);

                        this.with_lifetime_rib(
                            LifetimeRibKind::Elided(LifetimeRes::Static {
                                suppress_elision_warning: true,
                            }),
                            |this| this.visit_ty(ty),
                        );

                        if let Some(expr) = expr {
                            this.resolve_const_body(expr, Some((ident, ConstantItemKind::Const)));
                        }
                    },
                );
                self.resolve_define_opaques(define_opaque);
            }

            ItemKind::Use(ref use_tree) => {
                let maybe_exported = match use_tree.kind {
                    UseTreeKind::Simple(_) | UseTreeKind::Glob => MaybeExported::Ok(item.id),
                    UseTreeKind::Nested { .. } => MaybeExported::NestedUse(&item.vis),
                };
                self.resolve_doc_links(&item.attrs, maybe_exported);

                self.future_proof_import(use_tree);
            }

            ItemKind::MacroDef(_, ref macro_def) => {
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

            ItemKind::Delegation(ref delegation) => {
                let span = delegation.path.segments.last().unwrap().ident.span;
                self.with_generic_param_rib(
                    &[],
                    RibKind::Item(HasGenericParams::Yes(span), def_kind),
                    LifetimeRibKind::Generics {
                        binder: item.id,
                        kind: LifetimeBinderKind::Function,
                        span,
                    },
                    |this| this.resolve_delegation(delegation),
                );
            }

            ItemKind::ExternCrate(..) => {}

            ItemKind::MacCall(_) | ItemKind::DelegationMac(..) => {
                panic!("unexpanded macro in resolve!")
            }
        }
    }

    fn with_generic_param_rib<'c, F>(
        &'c mut self,
        params: &'c [GenericParam],
        kind: RibKind<'ra>,
        lifetime_kind: LifetimeRibKind,
        f: F,
    ) where
        F: FnOnce(&mut Self),
    {
        debug!("with_generic_param_rib");
        let LifetimeRibKind::Generics { binder, span: generics_span, kind: generics_kind, .. } =
            lifetime_kind
        else {
            panic!()
        };

        let mut function_type_rib = Rib::new(kind);
        let mut function_value_rib = Rib::new(kind);
        let mut function_lifetime_rib = LifetimeRib::new(lifetime_kind);

        // Only check for shadowed bindings if we're declaring new params.
        if !params.is_empty() {
            let mut seen_bindings = FxHashMap::default();
            // Store all seen lifetimes names from outer scopes.
            let mut seen_lifetimes = FxHashSet::default();

            // We also can't shadow bindings from associated parent items.
            for ns in [ValueNS, TypeNS] {
                for parent_rib in self.ribs[ns].iter().rev() {
                    // Break at mod level, to account for nested items which are
                    // allowed to shadow generic param names.
                    if matches!(parent_rib.kind, RibKind::Module(..)) {
                        break;
                    }

                    seen_bindings
                        .extend(parent_rib.bindings.keys().map(|ident| (*ident, ident.span)));
                }
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
                        let err = ResolutionError::NameAlreadyUsedInParameterList(ident, span);
                        self.report_error(param.ident.span, err);
                        let rib = match param.kind {
                            GenericParamKind::Lifetime => {
                                // Record lifetime res, so lowering knows there is something fishy.
                                self.record_lifetime_param(param.id, LifetimeRes::Error);
                                continue;
                            }
                            GenericParamKind::Type { .. } => &mut function_type_rib,
                            GenericParamKind::Const { .. } => &mut function_value_rib,
                        };

                        // Taint the resolution in case of errors to prevent follow up errors in typeck
                        self.r.record_partial_res(param.id, PartialRes::new(Res::Err));
                        rib.bindings.insert(ident, Res::Err);
                        continue;
                    }
                    Entry::Vacant(entry) => {
                        entry.insert(param.ident.span);
                    }
                }

                if param.ident.name == kw::UnderscoreLifetime {
                    self.r
                        .dcx()
                        .emit_err(errors::UnderscoreLifetimeIsReserved { span: param.ident.span });
                    // Record lifetime res, so lowering knows there is something fishy.
                    self.record_lifetime_param(param.id, LifetimeRes::Error);
                    continue;
                }

                if param.ident.name == kw::StaticLifetime {
                    self.r.dcx().emit_err(errors::StaticLifetimeIsReserved {
                        span: param.ident.span,
                        lifetime: param.ident,
                    });
                    // Record lifetime res, so lowering knows there is something fishy.
                    self.record_lifetime_param(param.id, LifetimeRes::Error);
                    continue;
                }

                let def_id = self.r.local_def_id(param.id);

                // Plain insert (no renaming).
                let (rib, def_kind) = match param.kind {
                    GenericParamKind::Type { .. } => (&mut function_type_rib, DefKind::TyParam),
                    GenericParamKind::Const { .. } => {
                        (&mut function_value_rib, DefKind::ConstParam)
                    }
                    GenericParamKind::Lifetime => {
                        let res = LifetimeRes::Param { param: def_id, binder };
                        self.record_lifetime_param(param.id, res);
                        function_lifetime_rib.bindings.insert(ident, (param.id, res));
                        continue;
                    }
                };

                let res = match kind {
                    RibKind::Item(..) | RibKind::AssocItem => {
                        Res::Def(def_kind, def_id.to_def_id())
                    }
                    RibKind::Normal => {
                        // FIXME(non_lifetime_binders): Stop special-casing
                        // const params to error out here.
                        if self.r.tcx.features().non_lifetime_binders()
                            && matches!(param.kind, GenericParamKind::Type { .. })
                        {
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

    fn with_label_rib(&mut self, kind: RibKind<'ra>, f: impl FnOnce(&mut Self)) {
        self.label_ribs.push(Rib::new(kind));
        f(self);
        self.label_ribs.pop();
    }

    fn with_static_rib(&mut self, def_kind: DefKind, f: impl FnOnce(&mut Self)) {
        let kind = RibKind::Item(HasGenericParams::No, def_kind);
        self.with_rib(ValueNS, kind, |this| this.with_rib(TypeNS, kind, f))
    }

    // HACK(min_const_generics, generic_const_exprs): We
    // want to keep allowing `[0; size_of::<*mut T>()]`
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
        let f = |this: &mut Self| {
            this.with_rib(ValueNS, RibKind::ConstantItem(may_use_generics, item), |this| {
                this.with_rib(
                    TypeNS,
                    RibKind::ConstantItem(
                        may_use_generics.force_yes_if(is_repeat == IsRepeatExpr::Yes),
                        item,
                    ),
                    |this| {
                        this.with_label_rib(RibKind::ConstantItem(may_use_generics, item), f);
                    },
                )
            })
        };

        if let ConstantHasGenerics::No(cause) = may_use_generics {
            self.with_lifetime_rib(LifetimeRibKind::ConcreteAnonConst(cause), f)
        } else {
            f(self)
        }
    }

    fn with_current_self_type<T>(&mut self, self_type: &Ty, f: impl FnOnce(&mut Self) -> T) -> T {
        // Handle nested impls (inside fn bodies)
        let previous_value =
            replace(&mut self.diag_metadata.current_self_type, Some(self_type.clone()));
        let result = f(self);
        self.diag_metadata.current_self_type = previous_value;
        result
    }

    fn with_current_self_item<T>(&mut self, self_item: &Item, f: impl FnOnce(&mut Self) -> T) -> T {
        let previous_value = replace(&mut self.diag_metadata.current_self_item, Some(self_item.id));
        let result = f(self);
        self.diag_metadata.current_self_item = previous_value;
        result
    }

    /// When evaluating a `trait` use its associated types' idents for suggestions in E0412.
    fn resolve_trait_items(&mut self, trait_items: &'ast [P<AssocItem>]) {
        let trait_assoc_items =
            replace(&mut self.diag_metadata.current_trait_assoc_items, Some(trait_items));

        let walk_assoc_item =
            |this: &mut Self, generics: &Generics, kind, item: &'ast AssocItem| {
                this.with_generic_param_rib(
                    &generics.params,
                    RibKind::AssocItem,
                    LifetimeRibKind::Generics { binder: item.id, span: generics.span, kind },
                    |this| visit::walk_assoc_item(this, item, AssocCtxt::Trait),
                );
            };

        for item in trait_items {
            self.resolve_doc_links(&item.attrs, MaybeExported::Ok(item.id));
            match &item.kind {
                AssocItemKind::Const(box ast::ConstItem {
                    generics,
                    ty,
                    expr,
                    define_opaque,
                    ..
                }) => {
                    self.with_generic_param_rib(
                        &generics.params,
                        RibKind::AssocItem,
                        LifetimeRibKind::Generics {
                            binder: item.id,
                            span: generics.span,
                            kind: LifetimeBinderKind::ConstItem,
                        },
                        |this| {
                            this.with_lifetime_rib(
                                LifetimeRibKind::StaticIfNoLifetimeInScope {
                                    lint_id: item.id,
                                    emit_lint: false,
                                },
                                |this| {
                                    this.visit_generics(generics);
                                    this.visit_ty(ty);

                                    // Only impose the restrictions of `ConstRibKind` for an
                                    // actual constant expression in a provided default.
                                    if let Some(expr) = expr {
                                        // We allow arbitrary const expressions inside of associated consts,
                                        // even if they are potentially not const evaluatable.
                                        //
                                        // Type parameters can already be used and as associated consts are
                                        // not used as part of the type system, this is far less surprising.
                                        this.resolve_const_body(expr, None);
                                    }
                                },
                            )
                        },
                    );

                    self.resolve_define_opaques(define_opaque);
                }
                AssocItemKind::Fn(box Fn { generics, define_opaque, .. }) => {
                    walk_assoc_item(self, generics, LifetimeBinderKind::Function, item);

                    self.resolve_define_opaques(define_opaque);
                }
                AssocItemKind::Delegation(delegation) => {
                    self.with_generic_param_rib(
                        &[],
                        RibKind::AssocItem,
                        LifetimeRibKind::Generics {
                            binder: item.id,
                            kind: LifetimeBinderKind::Function,
                            span: delegation.path.segments.last().unwrap().ident.span,
                        },
                        |this| this.resolve_delegation(delegation),
                    );
                }
                AssocItemKind::Type(box TyAlias { generics, .. }) => self
                    .with_lifetime_rib(LifetimeRibKind::AnonymousReportError, |this| {
                        walk_assoc_item(this, generics, LifetimeBinderKind::Item, item)
                    }),
                AssocItemKind::MacCall(_) | AssocItemKind::DelegationMac(..) => {
                    panic!("unexpanded macro in resolve!")
                }
            };
        }

        self.diag_metadata.current_trait_assoc_items = trait_assoc_items;
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
            self.diag_metadata.currently_processing_impl_trait =
                Some((trait_ref.clone(), self_type.clone()));
            let res = self.smart_resolve_path_fragment(
                &None,
                &path,
                PathSource::Trait(AliasPossibility::No),
                Finalize::new(trait_ref.ref_id, trait_ref.path.span),
                RecordPartialRes::Yes,
                None,
            );
            self.diag_metadata.currently_processing_impl_trait = None;
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
        let mut self_type_rib = Rib::new(RibKind::Normal);

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
            RibKind::Item(HasGenericParams::Yes(generics.span), self.r.local_def_kind(item_id)),
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
            AssocItemKind::Const(box ast::ConstItem {
                ident,
                generics,
                ty,
                expr,
                define_opaque,
                ..
            }) => {
                debug!("resolve_implementation AssocItemKind::Const");
                self.with_generic_param_rib(
                    &generics.params,
                    RibKind::AssocItem,
                    LifetimeRibKind::Generics {
                        binder: item.id,
                        span: generics.span,
                        kind: LifetimeBinderKind::ConstItem,
                    },
                    |this| {
                        this.with_lifetime_rib(
                            // Until these are a hard error, we need to create them within the correct binder,
                            // Otherwise the lifetimes of this assoc const think they are lifetimes of the trait.
                            LifetimeRibKind::AnonymousCreateParameter {
                                binder: item.id,
                                report_in_path: true,
                            },
                            |this| {
                                this.with_lifetime_rib(
                                    LifetimeRibKind::StaticIfNoLifetimeInScope {
                                        lint_id: item.id,
                                        // In impls, it's not a hard error yet due to backcompat.
                                        emit_lint: true,
                                    },
                                    |this| {
                                        // If this is a trait impl, ensure the const
                                        // exists in trait
                                        this.check_trait_item(
                                            item.id,
                                            *ident,
                                            &item.kind,
                                            ValueNS,
                                            item.span,
                                            seen_trait_items,
                                            |i, s, c| ConstNotMemberOfTrait(i, s, c),
                                        );

                                        this.visit_generics(generics);
                                        this.visit_ty(ty);
                                        if let Some(expr) = expr {
                                            // We allow arbitrary const expressions inside of associated consts,
                                            // even if they are potentially not const evaluatable.
                                            //
                                            // Type parameters can already be used and as associated consts are
                                            // not used as part of the type system, this is far less surprising.
                                            this.resolve_const_body(expr, None);
                                        }
                                    },
                                )
                            },
                        );
                    },
                );
                self.resolve_define_opaques(define_opaque);
            }
            AssocItemKind::Fn(box Fn { ident, generics, define_opaque, .. }) => {
                debug!("resolve_implementation AssocItemKind::Fn");
                // We also need a new scope for the impl item type parameters.
                self.with_generic_param_rib(
                    &generics.params,
                    RibKind::AssocItem,
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
                            *ident,
                            &item.kind,
                            ValueNS,
                            item.span,
                            seen_trait_items,
                            |i, s, c| MethodNotMemberOfTrait(i, s, c),
                        );

                        visit::walk_assoc_item(this, item, AssocCtxt::Impl { of_trait: true })
                    },
                );

                self.resolve_define_opaques(define_opaque);
            }
            AssocItemKind::Type(box TyAlias { ident, generics, .. }) => {
                self.diag_metadata.in_non_gat_assoc_type = Some(generics.params.is_empty());
                debug!("resolve_implementation AssocItemKind::Type");
                // We also need a new scope for the impl item type parameters.
                self.with_generic_param_rib(
                    &generics.params,
                    RibKind::AssocItem,
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
                                *ident,
                                &item.kind,
                                TypeNS,
                                item.span,
                                seen_trait_items,
                                |i, s, c| TypeNotMemberOfTrait(i, s, c),
                            );

                            visit::walk_assoc_item(this, item, AssocCtxt::Impl { of_trait: true })
                        });
                    },
                );
                self.diag_metadata.in_non_gat_assoc_type = None;
            }
            AssocItemKind::Delegation(box delegation) => {
                debug!("resolve_implementation AssocItemKind::Delegation");
                self.with_generic_param_rib(
                    &[],
                    RibKind::AssocItem,
                    LifetimeRibKind::Generics {
                        binder: item.id,
                        kind: LifetimeBinderKind::Function,
                        span: delegation.path.segments.last().unwrap().ident.span,
                    },
                    |this| {
                        this.check_trait_item(
                            item.id,
                            delegation.ident,
                            &item.kind,
                            ValueNS,
                            item.span,
                            seen_trait_items,
                            |i, s, c| MethodNotMemberOfTrait(i, s, c),
                        );

                        this.resolve_delegation(delegation)
                    },
                );
            }
            AssocItemKind::MacCall(_) | AssocItemKind::DelegationMac(..) => {
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
        F: FnOnce(Ident, String, Option<Symbol>) -> ResolutionError<'ra>,
    {
        // If there is a TraitRef in scope for an impl, then the method must be in the trait.
        let Some((module, _)) = self.current_trait_ref else {
            return;
        };
        ident.span.normalize_to_macros_2_0_and_adjust(module.expansion);
        let key = BindingKey::new(ident, ns);
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
            let key = BindingKey::new(ident, ns);
            binding = self.r.resolution(module, key).try_borrow().ok().and_then(|r| r.binding);
            debug!(?binding);
        }

        let feed_visibility = |this: &mut Self, def_id| {
            let vis = this.r.tcx.visibility(def_id);
            let vis = if vis.is_visible_locally() {
                vis.expect_local()
            } else {
                this.r.dcx().span_delayed_bug(
                    span,
                    "error should be emitted when an unexpected trait item is used",
                );
                rustc_middle::ty::Visibility::Public
            };
            this.r.feed_visibility(this.r.feed(id), vis);
        };

        let Some(binding) = binding else {
            // We could not find the method: report an error.
            let candidate = self.find_similarly_named_assoc_item(ident.name, kind);
            let path = &self.current_trait_ref.as_ref().unwrap().1.path;
            let path_names = path_names_to_string(path);
            self.report_error(span, err(ident, path_names, candidate));
            feed_visibility(self, module.def_id());
            return;
        };

        let res = binding.res();
        let Res::Def(def_kind, id_in_trait) = res else { bug!() };
        feed_visibility(self, id_in_trait);

        match seen_trait_items.entry(id_in_trait) {
            Entry::Occupied(entry) => {
                self.report_error(
                    span,
                    ResolutionError::TraitImplDuplicate {
                        name: ident,
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
            | (DefKind::AssocConst, AssocItemKind::Const(..))
            | (DefKind::AssocFn, AssocItemKind::Delegation(..)) => {
                self.r.record_partial_res(id, PartialRes::new(res));
                return;
            }
            _ => {}
        }

        // The method kind does not correspond to what appeared in the trait, report.
        let path = &self.current_trait_ref.as_ref().unwrap().1.path;
        let (code, kind) = match kind {
            AssocItemKind::Const(..) => (E0323, "const"),
            AssocItemKind::Fn(..) => (E0324, "method"),
            AssocItemKind::Type(..) => (E0325, "type"),
            AssocItemKind::Delegation(..) => (E0324, "method"),
            AssocItemKind::MacCall(..) | AssocItemKind::DelegationMac(..) => {
                span_bug!(span, "unexpanded macro")
            }
        };
        let trait_path = path_names_to_string(path);
        self.report_error(
            span,
            ResolutionError::TraitImplMismatch {
                name: ident,
                kind,
                code,
                trait_path,
                trait_item_span: binding.span,
            },
        );
    }

    fn resolve_const_body(&mut self, expr: &'ast Expr, item: Option<(Ident, ConstantItemKind)>) {
        self.with_lifetime_rib(LifetimeRibKind::Elided(LifetimeRes::Infer), |this| {
            this.with_constant_rib(IsRepeatExpr::No, ConstantHasGenerics::Yes, item, |this| {
                this.visit_expr(expr)
            });
        })
    }

    fn resolve_delegation(&mut self, delegation: &'ast Delegation) {
        self.smart_resolve_path(
            delegation.id,
            &delegation.qself,
            &delegation.path,
            PathSource::Delegation,
        );
        if let Some(qself) = &delegation.qself {
            self.visit_ty(&qself.ty);
        }
        self.visit_path(&delegation.path, delegation.id);
        let Some(body) = &delegation.body else { return };
        self.with_rib(ValueNS, RibKind::FnOrCoroutine, |this| {
            // `PatBoundCtx` is not necessary in this context
            let mut bindings = smallvec![(PatBoundCtx::Product, Default::default())];

            let span = delegation.path.segments.last().unwrap().ident.span;
            this.fresh_binding(
                Ident::new(kw::SelfLower, span),
                delegation.id,
                PatternSource::FnParam,
                &mut bindings,
            );
            this.visit_block(body);
        });
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
        visit_opt!(self, visit_ty, &local.ty);

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

    /// Build a map from pattern identifiers to binding-info's, and check the bindings are
    /// consistent when encountering or-patterns and never patterns.
    /// This is done hygienically: this could arise for a macro that expands into an or-pattern
    /// where one 'x' was from the user and one 'x' came from the macro.
    ///
    /// A never pattern by definition indicates an unreachable case. For example, matching on
    /// `Result<T, &!>` could look like:
    /// ```rust
    /// # #![feature(never_type)]
    /// # #![feature(never_patterns)]
    /// # fn bar(_x: u32) {}
    /// let foo: Result<u32, &!> = Ok(0);
    /// match foo {
    ///     Ok(x) => bar(x),
    ///     Err(&!),
    /// }
    /// ```
    /// This extends to product types: `(x, !)` is likewise unreachable. So it doesn't make sense to
    /// have a binding here, and we tell the user to use `_` instead.
    fn compute_and_check_binding_map(
        &mut self,
        pat: &Pat,
    ) -> Result<FxIndexMap<Ident, BindingInfo>, IsNeverPattern> {
        let mut binding_map = FxIndexMap::default();
        let mut is_never_pat = false;

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
                    match self.compute_and_check_or_pat_binding_map(ps) {
                        Ok(bm) => binding_map.extend(bm),
                        Err(IsNeverPattern) => is_never_pat = true,
                    }
                    return false;
                }
                PatKind::Never => is_never_pat = true,
                _ => {}
            }

            true
        });

        if is_never_pat {
            for (_, binding) in binding_map {
                self.report_error(binding.span, ResolutionError::BindingInNeverPattern);
            }
            Err(IsNeverPattern)
        } else {
            Ok(binding_map)
        }
    }

    fn is_base_res_local(&self, nid: NodeId) -> bool {
        matches!(
            self.r.partial_res_map.get(&nid).map(|res| res.expect_full_res()),
            Some(Res::Local(..))
        )
    }

    /// Compute the binding map for an or-pattern. Checks that all of the arms in the or-pattern
    /// have exactly the same set of bindings, with the same binding modes for each.
    /// Returns the computed binding map and a boolean indicating whether the pattern is a never
    /// pattern.
    ///
    /// A never pattern by definition indicates an unreachable case. For example, destructuring a
    /// `Result<T, &!>` could look like:
    /// ```rust
    /// # #![feature(never_type)]
    /// # #![feature(never_patterns)]
    /// # fn foo() -> Result<bool, &'static !> { Ok(true) }
    /// let (Ok(x) | Err(&!)) = foo();
    /// # let _ = x;
    /// ```
    /// Because the `Err(&!)` branch is never reached, it does not need to have the same bindings as
    /// the other branches of the or-pattern. So we must ignore never pattern when checking the
    /// bindings of an or-pattern.
    /// Moreover, if all the subpatterns are never patterns (e.g. `Ok(!) | Err(!)`), then the
    /// pattern as a whole counts as a never pattern (since it's definitionallly unreachable).
    fn compute_and_check_or_pat_binding_map(
        &mut self,
        pats: &[P<Pat>],
    ) -> Result<FxIndexMap<Ident, BindingInfo>, IsNeverPattern> {
        let mut missing_vars = FxIndexMap::default();
        let mut inconsistent_vars = FxIndexMap::default();

        // 1) Compute the binding maps of all arms; we must ignore never patterns here.
        let not_never_pats = pats
            .iter()
            .filter_map(|pat| {
                let binding_map = self.compute_and_check_binding_map(pat).ok()?;
                Some((binding_map, pat))
            })
            .collect::<Vec<_>>();

        // 2) Record any missing bindings or binding mode inconsistencies.
        for (map_outer, pat_outer) in not_never_pats.iter() {
            // Check against all arms except for the same pattern which is always self-consistent.
            let inners = not_never_pats
                .iter()
                .filter(|(_, pat)| pat.id != pat_outer.id)
                .flat_map(|(map, _)| map);

            for (&name, binding_inner) in inners {
                match map_outer.get(&name) {
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
        for (name, mut v) in missing_vars {
            if inconsistent_vars.contains_key(&name) {
                v.could_be_path = false;
            }
            self.report_error(
                *v.origin.iter().next().unwrap(),
                ResolutionError::VariableNotBoundInPattern(v, self.parent_scope),
            );
        }

        // 4) Report all inconsistencies in binding modes we found.
        for (name, v) in inconsistent_vars {
            self.report_error(v.0, ResolutionError::VariableBoundWithDifferentMode(name, v.1));
        }

        // 5) Bubble up the final binding map.
        if not_never_pats.is_empty() {
            // All the patterns are never patterns, so the whole or-pattern is one too.
            Err(IsNeverPattern)
        } else {
            let mut binding_map = FxIndexMap::default();
            for (bm, _) in not_never_pats {
                binding_map.extend(bm);
            }
            Ok(binding_map)
        }
    }

    /// Check the consistency of bindings wrt or-patterns and never patterns.
    fn check_consistent_bindings(&mut self, pat: &'ast Pat) {
        let mut is_or_or_never = false;
        pat.walk(&mut |pat| match pat.kind {
            PatKind::Or(..) | PatKind::Never => {
                is_or_or_never = true;
                false
            }
            _ => true,
        });
        if is_or_or_never {
            let _ = self.compute_and_check_binding_map(pat);
        }
    }

    fn resolve_arm(&mut self, arm: &'ast Arm) {
        self.with_rib(ValueNS, RibKind::Normal, |this| {
            this.resolve_pattern_top(&arm.pat, PatternSource::Match);
            visit_opt!(this, visit_expr, &arm.guard);
            visit_opt!(this, visit_expr, &arm.body);
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
        self.check_consistent_bindings(pat);
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
    #[tracing::instrument(skip(self, bindings), level = "debug")]
    fn resolve_pattern_inner(
        &mut self,
        pat: &Pat,
        pat_src: PatternSource,
        bindings: &mut SmallVec<[(PatBoundCtx, FxHashSet<Ident>); 1]>,
    ) {
        // Visit all direct subpatterns of this pattern.
        pat.walk(&mut |pat| {
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
                PatKind::Struct(ref qself, ref path, ref _fields, ref rest) => {
                    self.smart_resolve_path(pat.id, qself, path, PathSource::Struct);
                    self.record_patterns_with_skipped_bindings(pat, rest);
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

    fn record_patterns_with_skipped_bindings(&mut self, pat: &Pat, rest: &ast::PatFieldsRest) {
        match rest {
            ast::PatFieldsRest::Rest | ast::PatFieldsRest::Recovered(_) => {
                // Record that the pattern doesn't introduce all the bindings it could.
                if let Some(partial_res) = self.r.partial_res_map.get(&pat.id)
                    && let Some(res) = partial_res.full_res()
                    && let Some(def_id) = res.opt_def_id()
                {
                    self.ribs[ValueNS]
                        .last_mut()
                        .unwrap()
                        .patterns_with_skipped_bindings
                        .entry(def_id)
                        .or_default()
                        .push((
                            pat.span,
                            match rest {
                                ast::PatFieldsRest::Recovered(guar) => Err(*guar),
                                _ => Ok(()),
                            },
                        ));
                }
            }
            ast::PatFieldsRest::None => {}
        }
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
            self.report_error(ident.span, error(ident));
        }

        // Record as bound.
        bindings.last_mut().unwrap().1.insert(ident);

        if already_bound_or {
            // `Variant1(a) | Variant2(a)`, ok
            // Reuse definition from the first `a`.
            self.innermost_rib_bindings(ValueNS)[&ident]
        } else {
            // A completely fresh binding is added to the set.
            let res = Res::Local(pat_id);
            self.innermost_rib_bindings(ValueNS).insert(ident, res);
            res
        }
    }

    fn innermost_rib_bindings(&mut self, ns: Namespace) -> &mut FxIndexMap<Ident, Res> {
        &mut self.ribs[ns].last_mut().unwrap().bindings
    }

    fn try_resolve_as_non_binding(
        &mut self,
        pat_src: PatternSource,
        ann: BindingMode,
        ident: Ident,
        has_sub: bool,
    ) -> Option<Res> {
        // An immutable (no `mut`) by-value (no `ref`) binding pattern without
        // a sub pattern (no `@ $pat`) is syntactically ambiguous as it could
        // also be interpreted as a path to e.g. a constant, variant, etc.
        let is_syntactic_ambiguity = !has_sub && ann == BindingMode::NONE;

        let ls_binding = self.maybe_resolve_ident_in_lexical_scope(ident, ValueNS)?;
        let (res, binding) = match ls_binding {
            LexicalScopeBinding::Item(binding)
                if is_syntactic_ambiguity && binding.is_ambiguity_recursive() =>
            {
                // For ambiguous bindings we don't know all their definitions and cannot check
                // whether they can be shadowed by fresh bindings or not, so force an error.
                // issues/33118#issuecomment-233962221 (see below) still applies here,
                // but we have to ignore it for backward compatibility.
                self.r.record_use(ident, binding, Used::Other);
                return None;
            }
            LexicalScopeBinding::Item(binding) => (binding.res(), Some(binding)),
            LexicalScopeBinding::Res(res) => (res, None),
        };

        match res {
            Res::SelfCtor(_) // See #70549.
            | Res::Def(
                DefKind::Ctor(_, CtorKind::Const) | DefKind::Const | DefKind::AssocConst | DefKind::ConstParam,
                _,
            ) if is_syntactic_ambiguity => {
                // Disambiguate in favor of a unit struct/variant or constant pattern.
                if let Some(binding) = binding {
                    self.r.record_use(ident, binding, Used::Other);
                }
                Some(res)
            }
            Res::Def(DefKind::Ctor(..) | DefKind::Const | DefKind::AssocConst | DefKind::Static { .. }, _) => {
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
            Res::Def(DefKind::Fn | DefKind::AssocFn, _) | Res::Local(..) | Res::Err => {
                // These entities are explicitly allowed to be shadowed by fresh bindings.
                None
            }
            Res::SelfCtor(_) => {
                // We resolve `Self` in pattern position as an ident sometimes during recovery,
                // so delay a bug instead of ICEing.
                self.r.dcx().span_delayed_bug(
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
            RecordPartialRes::Yes,
            None,
        );
    }

    #[instrument(level = "debug", skip(self))]
    fn smart_resolve_path_fragment(
        &mut self,
        qself: &Option<P<QSelf>>,
        path: &[Segment],
        source: PathSource<'ast>,
        finalize: Finalize,
        record_partial_res: RecordPartialRes,
        parent_qself: Option<&QSelf>,
    ) -> PartialRes {
        let ns = source.namespace();

        let Finalize { node_id, path_span, .. } = finalize;
        let report_errors = |this: &mut Self, res: Option<Res>| {
            if this.should_report_errs() {
                let (err, candidates) = this.smart_resolve_report_errors(
                    path,
                    None,
                    path_span,
                    source,
                    res,
                    parent_qself,
                );

                let def_id = this.parent_scope.module.nearest_parent_mod();
                let instead = res.is_some();
                let suggestion = if let Some((start, end)) = this.diag_metadata.in_range
                    && path[0].ident.span.lo() == end.span.lo()
                    && !matches!(start.kind, ExprKind::Lit(_))
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
                } else if res.is_none()
                    && let PathSource::Type
                    | PathSource::Expr(_)
                    | PathSource::PreciseCapturingArg(..) = source
                {
                    this.suggest_adding_generic_parameter(path, source)
                } else {
                    None
                };

                let ue = UseError {
                    err,
                    candidates,
                    def_id,
                    instead,
                    suggestion,
                    path: path.into(),
                    is_call: source.is_call(),
                };

                this.r.use_injections.push(ue);
            }

            PartialRes::new(Res::Err)
        };

        // For paths originating from calls (like in `HashMap::new()`), tries
        // to enrich the plain `failed to resolve: ...` message with hints
        // about possible missing imports.
        //
        // Similar thing, for types, happens in `report_errors` above.
        let report_errors_for_call =
            |this: &mut Self, parent_err: Spanned<ResolutionError<'ra>>| {
                // Before we start looking for candidates, we have to get our hands
                // on the type user is trying to perform invocation on; basically:
                // we're transforming `HashMap::new` into just `HashMap`.
                let (following_seg, prefix_path) = match path.split_last() {
                    Some((last, path)) if !path.is_empty() => (Some(last), path),
                    _ => return Some(parent_err),
                };

                let (mut err, candidates) = this.smart_resolve_report_errors(
                    prefix_path,
                    following_seg,
                    path_span,
                    PathSource::Type,
                    None,
                    parent_qself,
                );

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
                err.messages = take(&mut parent_err.messages);
                err.code = take(&mut parent_err.code);
                swap(&mut err.span, &mut parent_err.span);
                err.children = take(&mut parent_err.children);
                err.sort_span = parent_err.sort_span;
                err.is_lint = parent_err.is_lint.clone();

                // merge the parent_err's suggestions with the typo (err's) suggestions
                match &mut err.suggestions {
                    Suggestions::Enabled(typo_suggestions) => match &mut parent_err.suggestions {
                        Suggestions::Enabled(parent_suggestions) => {
                            // If both suggestions are enabled, append parent_err's suggestions to err's suggestions.
                            typo_suggestions.append(parent_suggestions)
                        }
                        Suggestions::Sealed(_) | Suggestions::Disabled => {
                            // If the parent's suggestions are either sealed or disabled, it signifies that
                            // new suggestions cannot be added or removed from the diagnostic. Therefore,
                            // we assign both types of suggestions to err's suggestions and discard the
                            // existing suggestions in err.
                            err.suggestions = std::mem::take(&mut parent_err.suggestions);
                        }
                    },
                    Suggestions::Sealed(_) | Suggestions::Disabled => (),
                }

                parent_err.cancel();

                let def_id = this.parent_scope.module.nearest_parent_mod();

                if this.should_report_errs() {
                    if candidates.is_empty() {
                        if path.len() == 2
                            && let [segment] = prefix_path
                        {
                            // Delay to check whether methond name is an associated function or not
                            // ```
                            // let foo = Foo {};
                            // foo::bar(); // possibly suggest to foo.bar();
                            //```
                            err.stash(segment.ident.span, rustc_errors::StashKey::CallAssocMethod);
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
                // if we also have an associated type that matches the ident, stash a suggestion
                if let Some(items) = self.diag_metadata.current_trait_assoc_items
                    && let [Segment { ident, .. }] = path
                    && items.iter().any(|item| {
                        if let AssocItemKind::Type(alias) = &item.kind
                            && alias.ident == *ident
                        {
                            true
                        } else {
                            false
                        }
                    })
                {
                    let mut diag = self.r.tcx.dcx().struct_allow("");
                    diag.span_suggestion_verbose(
                        path_span.shrink_to_lo(),
                        "there is an associated type with the same name",
                        "Self::",
                        Applicability::MaybeIncorrect,
                    );
                    diag.stash(path_span, StashKey::AssociatedTypeSuggestion);
                }

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

        if record_partial_res == RecordPartialRes::Yes {
            // Avoid recording definition of `A::B` in `<T as A>::B::C`.
            self.r.record_partial_res(node_id, partial_res);
            self.resolve_elided_lifetimes_in_path(partial_res, path, source, path_span);
            self.lint_unused_qualifications(path, ns, finalize);
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
    fn report_error(&mut self, span: Span, resolution_error: ResolutionError<'ra>) {
        if self.should_report_errs() {
            self.r.report_error(span, resolution_error);
        }
    }

    #[inline]
    /// If we're actually rustdoc then avoid giving a name resolution error for `cfg()` items or
    // an invalid `use foo::*;` was found, which can cause unbounded amounts of "item not found"
    // errors. We silence them all.
    fn should_report_errs(&self) -> bool {
        !(self.r.tcx.sess.opts.actually_rustdoc && self.in_func_body)
            && !self.r.glob_error.is_some()
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
    ) -> Result<Option<PartialRes>, Spanned<ResolutionError<'ra>>> {
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
                self.r.resolve_macro_path(&path, None, &self.parent_scope, false, false, None)
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
    ) -> Result<Option<PartialRes>, Spanned<ResolutionError<'ra>>> {
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

            let num_privacy_errors = self.r.privacy_errors.len();
            // Make sure that `A` in `<T as A>::B::C` is a trait.
            let trait_res = self.smart_resolve_path_fragment(
                &None,
                &path[..qself.position],
                PathSource::Trait(AliasPossibility::No),
                Finalize::new(finalize.node_id, qself.path_span),
                RecordPartialRes::No,
                Some(&qself),
            );

            if trait_res.expect_full_res() == Res::Err {
                return Ok(Some(trait_res));
            }

            // Truncate additional privacy errors reported above,
            // because they'll be recomputed below.
            self.r.privacy_errors.truncate(num_privacy_errors);

            // Make sure `A::B` in `<T as A>::B::C` is a trait item.
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
                RecordPartialRes::No,
                Some(&qself),
            );

            // The remaining segments (the `C` in our example) will
            // have to be resolved by type-check, since that requires doing
            // trait resolution.
            return Ok(Some(PartialRes::with_unresolved_segments(
                partial_res.base_res(),
                partial_res.unresolved_segments() + path.len() - qself.position - 1,
            )));
        }

        let result = match self.resolve_path(path, Some(ns), Some(finalize)) {
            PathResult::NonModule(path_res) => path_res,
            PathResult::Module(ModuleOrUniformRoot::Module(module)) if !module.is_normal() => {
                PartialRes::new(module.res().unwrap())
            }
            // A part of this path references a `mod` that had a parse error. To avoid resolution
            // errors for each reference to that module, we don't emit an error for them until the
            // `mod` is fixed. this can have a significant cascade effect.
            PathResult::Failed { error_implied_by_parse_error: true, .. } => {
                PartialRes::new(Res::Err)
            }
            // In `a(::assoc_item)*` `a` cannot be a module. If `a` does resolve to a module we
            // don't report an error right away, but try to fallback to a primitive type.
            // So, we are still able to successfully resolve something like
            //
            // use std::u8; // bring module u8 in scope
            // fn f() -> u8 { // OK, resolves to primitive u8, not to std::u8
            //     u8::max_value() // OK, resolves to associated function <u8>::max_value,
            //                     // not to nonexistent std::u8::max_value
            // }
            //
            // Such behavior is required for backward compatibility.
            // The same fallback is used when `a` resolves to nothing.
            PathResult::Module(ModuleOrUniformRoot::Module(_)) | PathResult::Failed { .. }
                if (ns == TypeNS || path.len() > 1)
                    && PrimTy::from_name(path[0].ident.name).is_some() =>
            {
                let prim = PrimTy::from_name(path[0].ident.name).unwrap();
                let tcx = self.r.tcx();

                let gate_err_sym_msg = match prim {
                    PrimTy::Float(FloatTy::F16) if !tcx.features().f16() => {
                        Some((sym::f16, "the type `f16` is unstable"))
                    }
                    PrimTy::Float(FloatTy::F128) if !tcx.features().f128() => {
                        Some((sym::f128, "the type `f128` is unstable"))
                    }
                    _ => None,
                };

                if let Some((sym, msg)) = gate_err_sym_msg {
                    let span = path[0].ident.span;
                    if !span.allows_unstable(sym) {
                        feature_err(tcx.sess, sym, span, msg).emit();
                    }
                };

                PartialRes::with_unresolved_segments(Res::PrimTy(prim), path.len() - 1)
            }
            PathResult::Module(ModuleOrUniformRoot::Module(module)) => {
                PartialRes::new(module.res().unwrap())
            }
            PathResult::Failed {
                is_error_from_last_segment: false,
                span,
                label,
                suggestion,
                module,
                segment_name,
                error_implied_by_parse_error: _,
            } => {
                return Err(respan(
                    span,
                    ResolutionError::FailedToResolve {
                        segment: Some(segment_name),
                        label,
                        suggestion,
                        module,
                    },
                ));
            }
            PathResult::Module(..) | PathResult::Failed { .. } => return Ok(None),
            PathResult::Indeterminate => bug!("indeterminate path result in resolve_qpath"),
        };

        Ok(Some(result))
    }

    fn with_resolved_label(&mut self, label: Option<Label>, id: NodeId, f: impl FnOnce(&mut Self)) {
        if let Some(label) = label {
            if label.ident.as_str().as_bytes()[1] != b'_' {
                self.diag_metadata.unused_labels.insert(id, label.ident.span);
            }

            if let Ok((_, orig_span)) = self.resolve_label(label.ident) {
                diagnostics::signal_label_shadowing(self.r.tcx.sess, orig_span, label.ident)
            }

            self.with_label_rib(RibKind::Normal, |this| {
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
            self.ribs[ValueNS].push(Rib::new(RibKind::Module(anonymous_module)));
            self.ribs[TypeNS].push(Rib::new(RibKind::Module(anonymous_module)));
            self.parent_scope.module = anonymous_module;
        } else {
            self.ribs[ValueNS].push(Rib::new(RibKind::Normal));
        }

        // Descend into the block.
        for stmt in &block.stmts {
            if let StmtKind::Item(ref item) = stmt.kind
                && let ItemKind::MacroDef(..) = item.kind
            {
                num_macro_definition_ribs += 1;
                let res = self.r.local_def_id(item.id).to_def_id();
                self.ribs[ValueNS].push(Rib::new(RibKind::MacroDefinition(res)));
                self.label_ribs.push(Rib::new(RibKind::MacroDefinition(res)));
            }

            self.visit_stmt(stmt);
        }

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

    fn resolve_anon_const(&mut self, constant: &'ast AnonConst, anon_const_kind: AnonConstKind) {
        debug!(
            "resolve_anon_const(constant: {:?}, anon_const_kind: {:?})",
            constant, anon_const_kind
        );

        let is_trivial_const_arg = constant
            .value
            .is_potential_trivial_const_arg(self.r.tcx.features().min_generic_const_args());
        self.resolve_anon_const_manual(is_trivial_const_arg, anon_const_kind, |this| {
            this.resolve_expr(&constant.value, None)
        })
    }

    /// There are a few places that we need to resolve an anon const but we did not parse an
    /// anon const so cannot provide an `&'ast AnonConst`. Right now this is just unbraced
    /// const arguments that were parsed as type arguments, and `legacy_const_generics` which
    /// parse as normal function argument expressions. To avoid duplicating the code for resolving
    /// an anon const we have this function which lets the caller manually call `resolve_expr` or
    /// `smart_resolve_path`.
    fn resolve_anon_const_manual(
        &mut self,
        is_trivial_const_arg: bool,
        anon_const_kind: AnonConstKind,
        resolve_expr: impl FnOnce(&mut Self),
    ) {
        let is_repeat_expr = match anon_const_kind {
            AnonConstKind::ConstArg(is_repeat_expr) => is_repeat_expr,
            _ => IsRepeatExpr::No,
        };

        let may_use_generics = match anon_const_kind {
            AnonConstKind::EnumDiscriminant => {
                ConstantHasGenerics::No(NoConstantGenericsReason::IsEnumDiscriminant)
            }
            AnonConstKind::FieldDefaultValue => ConstantHasGenerics::Yes,
            AnonConstKind::InlineConst => ConstantHasGenerics::Yes,
            AnonConstKind::ConstArg(_) => {
                if self.r.tcx.features().generic_const_exprs() || is_trivial_const_arg {
                    ConstantHasGenerics::Yes
                } else {
                    ConstantHasGenerics::No(NoConstantGenericsReason::NonTrivialConstArg)
                }
            }
        };

        self.with_constant_rib(is_repeat_expr, may_use_generics, None, |this| {
            this.with_lifetime_rib(LifetimeRibKind::Elided(LifetimeRes::Infer), |this| {
                resolve_expr(this);
            });
        });
    }

    fn resolve_expr_field(&mut self, f: &'ast ExprField, e: &'ast Expr) {
        self.resolve_expr(&f.expr, Some(e));
        self.visit_ident(&f.ident);
        walk_list!(self, visit_attribute, f.attrs.iter());
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
                // This is the same as `visit::walk_expr(self, expr);`, but we want to pass the
                // parent in for accurate suggestions when encountering `Foo { bar }` that should
                // have been `Foo { bar: self.bar }`.
                if let Some(qself) = &se.qself {
                    self.visit_ty(&qself.ty);
                }
                self.visit_path(&se.path, expr.id);
                walk_list!(self, resolve_expr_field, &se.fields, expr);
                match &se.rest {
                    StructRest::Base(expr) => self.visit_expr(expr),
                    StructRest::Rest(_span) => {}
                    StructRest::None => {}
                }
            }

            ExprKind::Break(Some(label), _) | ExprKind::Continue(Some(label)) => {
                match self.resolve_label(label.ident) {
                    Ok((node_id, _)) => {
                        // Since this res is a label, it is never read.
                        self.r.label_res_map.insert(expr.id, node_id);
                        self.diag_metadata.unused_labels.swap_remove(&node_id);
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
                self.resolve_expr(e, Some(expr));
            }

            ExprKind::Let(ref pat, ref scrutinee, _, _) => {
                self.visit_expr(scrutinee);
                self.resolve_pattern_top(pat, PatternSource::Let);
            }

            ExprKind::If(ref cond, ref then, ref opt_else) => {
                self.with_rib(ValueNS, RibKind::Normal, |this| {
                    let old = this.diag_metadata.in_if_condition.replace(cond);
                    this.visit_expr(cond);
                    this.diag_metadata.in_if_condition = old;
                    this.visit_block(then);
                });
                if let Some(expr) = opt_else {
                    self.visit_expr(expr);
                }
            }

            ExprKind::Loop(ref block, label, _) => {
                self.resolve_labeled_block(label, expr.id, block)
            }

            ExprKind::While(ref cond, ref block, label) => {
                self.with_resolved_label(label, expr.id, |this| {
                    this.with_rib(ValueNS, RibKind::Normal, |this| {
                        let old = this.diag_metadata.in_if_condition.replace(cond);
                        this.visit_expr(cond);
                        this.diag_metadata.in_if_condition = old;
                        this.visit_block(block);
                    })
                });
            }

            ExprKind::ForLoop { ref pat, ref iter, ref body, label, kind: _ } => {
                self.visit_expr(iter);
                self.with_rib(ValueNS, RibKind::Normal, |this| {
                    this.resolve_pattern_top(pat, PatternSource::For);
                    this.resolve_labeled_block(label, expr.id, body);
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
                        let is_trivial_const_arg = argument.is_potential_trivial_const_arg(
                            self.r.tcx.features().min_generic_const_args(),
                        );
                        self.resolve_anon_const_manual(
                            is_trivial_const_arg,
                            AnonConstKind::ConstArg(IsRepeatExpr::No),
                            |this| this.resolve_expr(argument, None),
                        );
                    } else {
                        self.resolve_expr(argument, None);
                    }
                }
            }
            ExprKind::Type(ref _type_expr, ref _ty) => {
                visit::walk_expr(self, expr);
            }
            // For closures, RibKind::FnOrCoroutine is added in visit_fn
            ExprKind::Closure(box ast::Closure {
                binder: ClosureBinder::For { ref generic_params, span },
                ..
            }) => {
                self.with_generic_param_rib(
                    generic_params,
                    RibKind::Normal,
                    LifetimeRibKind::Generics {
                        binder: expr.id,
                        kind: LifetimeBinderKind::Closure,
                        span,
                    },
                    |this| visit::walk_expr(this, expr),
                );
            }
            ExprKind::Closure(..) => visit::walk_expr(self, expr),
            ExprKind::Gen(..) => {
                self.with_label_rib(RibKind::FnOrCoroutine, |this| visit::walk_expr(this, expr));
            }
            ExprKind::Repeat(ref elem, ref ct) => {
                self.visit_expr(elem);
                self.resolve_anon_const(ct, AnonConstKind::ConstArg(IsRepeatExpr::Yes));
            }
            ExprKind::ConstBlock(ref ct) => {
                self.resolve_anon_const(ct, AnonConstKind::InlineConst);
            }
            ExprKind::Index(ref elem, ref idx, _) => {
                self.resolve_expr(elem, Some(expr));
                self.visit_expr(idx);
            }
            ExprKind::Assign(ref lhs, ref rhs, _) => {
                if !self.diag_metadata.is_assign_rhs {
                    self.diag_metadata.in_assignment = Some(expr);
                }
                self.visit_expr(lhs);
                self.diag_metadata.is_assign_rhs = true;
                self.diag_metadata.in_assignment = None;
                self.visit_expr(rhs);
                self.diag_metadata.is_assign_rhs = false;
            }
            ExprKind::Range(Some(ref start), Some(ref end), RangeLimits::HalfOpen) => {
                self.diag_metadata.in_range = Some((start, end));
                self.resolve_expr(start, Some(expr));
                self.resolve_expr(end, Some(expr));
                self.diag_metadata.in_range = None;
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

    fn resolve_and_cache_rustdoc_path(&mut self, path_str: &str, ns: Namespace) -> Option<Res> {
        // FIXME: This caching may be incorrect in case of multiple `macro_rules`
        // items with the same name in the same module.
        // Also hygiene is not considered.
        let mut doc_link_resolutions = std::mem::take(&mut self.r.doc_link_resolutions);
        let res = *doc_link_resolutions
            .entry(self.parent_scope.module.nearest_parent_mod().expect_local())
            .or_default()
            .entry((Symbol::intern(path_str), ns))
            .or_insert_with_key(|(path, ns)| {
                let res = self.r.resolve_rustdoc_path(path.as_str(), *ns, self.parent_scope);
                if let Some(res) = res
                    && let Some(def_id) = res.opt_def_id()
                    && self.is_invalid_proc_macro_item_for_doc(def_id)
                {
                    // Encoding def ids in proc macro crate metadata will ICE,
                    // because it will only store proc macros for it.
                    return None;
                }
                res
            });
        self.r.doc_link_resolutions = doc_link_resolutions;
        res
    }

    fn is_invalid_proc_macro_item_for_doc(&self, did: DefId) -> bool {
        if !matches!(self.r.tcx.sess.opts.resolve_doc_links, ResolveDocLinks::ExportedMetadata)
            || !self.r.tcx.crate_types().contains(&CrateType::ProcMacro)
        {
            return false;
        }
        let Some(local_did) = did.as_local() else { return true };
        let Some(node_id) = self.r.def_id_to_node_id.get(local_did) else { return true };
        !self.r.proc_macros.contains(node_id)
    }

    fn resolve_doc_links(&mut self, attrs: &[Attribute], maybe_exported: MaybeExported<'_>) {
        match self.r.tcx.sess.opts.resolve_doc_links {
            ResolveDocLinks::None => return,
            ResolveDocLinks::ExportedMetadata
                if !self.r.tcx.crate_types().iter().copied().any(CrateType::has_metadata)
                    || !maybe_exported.eval(self.r) =>
            {
                return;
            }
            ResolveDocLinks::Exported
                if !maybe_exported.eval(self.r)
                    && !rustdoc::has_primitive_or_keyword_docs(attrs) =>
            {
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
                if let Some(res) = self.resolve_and_cache_rustdoc_path(&path_str, ns) {
                    // Rustdoc ignores tool attribute resolutions and attempts
                    // to resolve their prefixes for diagnostics.
                    any_resolved = !matches!(res, Res::NonMacroAttr(NonMacroAttrKind::Tool));
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
                            if self.is_invalid_proc_macro_item_for_doc(tr.def_id) {
                                // Encoding def ids in proc macro crate metadata will ICE.
                                // because it will only store proc macros for it.
                                return None;
                            }
                            Some(tr.def_id)
                        })
                        .collect()
                });
            self.r.doc_link_traits_in_scope = doc_link_traits_in_scope;
        }
    }

    fn lint_unused_qualifications(&mut self, path: &[Segment], ns: Namespace, finalize: Finalize) {
        // Don't lint on global paths because the user explicitly wrote out the full path.
        if let Some(seg) = path.first()
            && seg.ident.name == kw::PathRoot
        {
            return;
        }

        if finalize.path_span.from_expansion()
            || path.iter().any(|seg| seg.ident.span.from_expansion())
        {
            return;
        }

        let end_pos =
            path.iter().position(|seg| seg.has_generic_args).map_or(path.len(), |pos| pos + 1);
        let unqualified = path[..end_pos].iter().enumerate().skip(1).rev().find_map(|(i, seg)| {
            // Preserve the current namespace for the final path segment, but use the type
            // namespace for all preceding segments
            //
            // e.g. for `std::env::args` check the `ValueNS` for `args` but the `TypeNS` for
            // `std` and `env`
            //
            // If the final path segment is beyond `end_pos` all the segments to check will
            // use the type namespace
            let ns = if i + 1 == path.len() { ns } else { TypeNS };
            let res = self.r.partial_res_map.get(&seg.id?)?.full_res()?;
            let binding = self.resolve_ident_in_lexical_scope(seg.ident, ns, None, None)?;
            (res == binding.res()).then_some((seg, binding))
        });

        if let Some((seg, binding)) = unqualified {
            self.r.potentially_unnecessary_qualifications.push(UnnecessaryQualification {
                binding,
                node_id: finalize.node_id,
                path_span: finalize.path_span,
                removal_span: path[0].ident.span.until(seg.ident.span),
            });
        }
    }

    fn resolve_define_opaques(&mut self, define_opaque: &Option<ThinVec<(NodeId, Path)>>) {
        if let Some(define_opaque) = define_opaque {
            for (id, path) in define_opaque {
                self.smart_resolve_path(*id, &None, path, PathSource::DefineOpaques);
            }
        }
    }
}

/// Walks the whole crate in DFS order, visiting each item, counting the declared number of
/// lifetime generic parameters and function parameters.
struct ItemInfoCollector<'a, 'ra, 'tcx> {
    r: &'a mut Resolver<'ra, 'tcx>,
}

impl ItemInfoCollector<'_, '_, '_> {
    fn collect_fn_info(
        &mut self,
        header: FnHeader,
        decl: &FnDecl,
        id: NodeId,
        attrs: &[Attribute],
    ) {
        let sig = DelegationFnSig {
            header,
            param_count: decl.inputs.len(),
            has_self: decl.has_self(),
            c_variadic: decl.c_variadic(),
            target_feature: attrs.iter().any(|attr| attr.has_name(sym::target_feature)),
        };
        self.r.delegation_fn_sigs.insert(self.r.local_def_id(id), sig);
    }
}

impl<'ast> Visitor<'ast> for ItemInfoCollector<'_, '_, '_> {
    fn visit_item(&mut self, item: &'ast Item) {
        match &item.kind {
            ItemKind::TyAlias(box TyAlias { generics, .. })
            | ItemKind::Const(box ConstItem { generics, .. })
            | ItemKind::Fn(box Fn { generics, .. })
            | ItemKind::Enum(_, _, generics)
            | ItemKind::Struct(_, _, generics)
            | ItemKind::Union(_, _, generics)
            | ItemKind::Impl(box Impl { generics, .. })
            | ItemKind::Trait(box Trait { generics, .. })
            | ItemKind::TraitAlias(_, generics, _) => {
                if let ItemKind::Fn(box Fn { sig, .. }) = &item.kind {
                    self.collect_fn_info(sig.header, &sig.decl, item.id, &item.attrs);
                }

                let def_id = self.r.local_def_id(item.id);
                let count = generics
                    .params
                    .iter()
                    .filter(|param| matches!(param.kind, ast::GenericParamKind::Lifetime { .. }))
                    .count();
                self.r.item_generics_num_lifetimes.insert(def_id, count);
            }

            ItemKind::ForeignMod(ForeignMod { extern_span, safety: _, abi, items }) => {
                for foreign_item in items {
                    if let ForeignItemKind::Fn(box Fn { sig, .. }) = &foreign_item.kind {
                        let new_header =
                            FnHeader { ext: Extern::from_abi(*abi, *extern_span), ..sig.header };
                        self.collect_fn_info(new_header, &sig.decl, foreign_item.id, &item.attrs);
                    }
                }
            }

            ItemKind::Mod(..)
            | ItemKind::Static(..)
            | ItemKind::Use(..)
            | ItemKind::ExternCrate(..)
            | ItemKind::MacroDef(..)
            | ItemKind::GlobalAsm(..)
            | ItemKind::MacCall(..)
            | ItemKind::DelegationMac(..) => {}
            ItemKind::Delegation(..) => {
                // Delegated functions have lifetimes, their count is not necessarily zero.
                // But skipping the delegation items here doesn't mean that the count will be considered zero,
                // it means there will be a panic when retrieving the count,
                // but for delegation items we are never actually retrieving that count in practice.
            }
        }
        visit::walk_item(self, item)
    }

    fn visit_assoc_item(&mut self, item: &'ast AssocItem, ctxt: AssocCtxt) {
        if let AssocItemKind::Fn(box Fn { sig, .. }) = &item.kind {
            self.collect_fn_info(sig.header, &sig.decl, item.id, &item.attrs);
        }
        visit::walk_assoc_item(self, item, ctxt);
    }
}

impl<'ra, 'tcx> Resolver<'ra, 'tcx> {
    pub(crate) fn late_resolve_crate(&mut self, krate: &Crate) {
        visit::walk_crate(&mut ItemInfoCollector { r: self }, krate);
        let mut late_resolution_visitor = LateResolutionVisitor::new(self);
        late_resolution_visitor.resolve_doc_links(&krate.attrs, MaybeExported::Ok(CRATE_NODE_ID));
        visit::walk_crate(&mut late_resolution_visitor, krate);
        for (id, span) in late_resolution_visitor.diag_metadata.unused_labels.iter() {
            self.lint_buffer.buffer_lint(
                lint::builtin::UNUSED_LABELS,
                *id,
                *span,
                BuiltinLintDiag::UnusedLabel,
            );
        }
    }
}

/// Check if definition matches a path
fn def_id_matches_path(tcx: TyCtxt<'_>, mut def_id: DefId, expected_path: &[&str]) -> bool {
    let mut path = expected_path.iter().rev();
    while let (Some(parent), Some(next_step)) = (tcx.opt_parent(def_id), path.next()) {
        if !tcx.opt_item_name(def_id).is_some_and(|n| n.as_str() == *next_step) {
            return false;
        }
        def_id = parent;
    }
    true
}
