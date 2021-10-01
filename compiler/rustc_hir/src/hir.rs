use crate::def::{CtorKind, DefKind, Res};
use crate::def_id::{DefId, CRATE_DEF_ID};
crate use crate::hir_id::{HirId, ItemLocalId};
use crate::LangItem;

use rustc_ast::util::parser::ExprPrecedence;
use rustc_ast::{self as ast, CrateSugar, LlvmAsmDialect};
use rustc_ast::{Attribute, FloatTy, IntTy, Label, LitKind, StrStyle, TraitObjectSyntax, UintTy};
pub use rustc_ast::{BorrowKind, ImplPolarity, IsAuto};
pub use rustc_ast::{CaptureBy, Movability, Mutability};
use rustc_ast::{InlineAsmOptions, InlineAsmTemplatePiece};
use rustc_data_structures::fx::FxHashMap;
use rustc_index::vec::IndexVec;
use rustc_macros::HashStable_Generic;
use rustc_span::source_map::Spanned;
use rustc_span::symbol::{kw, sym, Ident, Symbol};
use rustc_span::{def_id::LocalDefId, BytePos};
use rustc_span::{MultiSpan, Span, DUMMY_SP};
use rustc_target::asm::InlineAsmRegOrRegClass;
use rustc_target::spec::abi::Abi;

use smallvec::SmallVec;
use std::collections::BTreeMap;
use std::fmt;

#[derive(Copy, Clone, Encodable, HashStable_Generic)]
pub struct Lifetime {
    pub hir_id: HirId,
    pub span: Span,

    /// Either "`'a`", referring to a named lifetime definition,
    /// or "``" (i.e., `kw::Empty`), for elision placeholders.
    ///
    /// HIR lowering inserts these placeholders in type paths that
    /// refer to type definitions needing lifetime parameters,
    /// `&T` and `&mut T`, and trait objects without `... + 'a`.
    pub name: LifetimeName,
}

#[derive(Debug, Clone, PartialEq, Eq, Encodable, Hash, Copy)]
#[derive(HashStable_Generic)]
pub enum ParamName {
    /// Some user-given name like `T` or `'x`.
    Plain(Ident),

    /// Synthetic name generated when user elided a lifetime in an impl header.
    ///
    /// E.g., the lifetimes in cases like these:
    ///
    ///     impl Foo for &u32
    ///     impl Foo<'_> for u32
    ///
    /// in that case, we rewrite to
    ///
    ///     impl<'f> Foo for &'f u32
    ///     impl<'f> Foo<'f> for u32
    ///
    /// where `'f` is something like `Fresh(0)`. The indices are
    /// unique per impl, but not necessarily continuous.
    Fresh(usize),

    /// Indicates an illegal name was given and an error has been
    /// reported (so we should squelch other derived errors). Occurs
    /// when, e.g., `'_` is used in the wrong place.
    Error,
}

impl ParamName {
    pub fn ident(&self) -> Ident {
        match *self {
            ParamName::Plain(ident) => ident,
            ParamName::Fresh(_) | ParamName::Error => {
                Ident::with_dummy_span(kw::UnderscoreLifetime)
            }
        }
    }

    pub fn normalize_to_macros_2_0(&self) -> ParamName {
        match *self {
            ParamName::Plain(ident) => ParamName::Plain(ident.normalize_to_macros_2_0()),
            param_name => param_name,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Encodable, Hash, Copy)]
#[derive(HashStable_Generic)]
pub enum LifetimeName {
    /// User-given names or fresh (synthetic) names.
    Param(ParamName),

    /// User wrote nothing (e.g., the lifetime in `&u32`).
    Implicit,

    /// Implicit lifetime in a context like `dyn Foo`. This is
    /// distinguished from implicit lifetimes elsewhere because the
    /// lifetime that they default to must appear elsewhere within the
    /// enclosing type.  This means that, in an `impl Trait` context, we
    /// don't have to create a parameter for them. That is, `impl
    /// Trait<Item = &u32>` expands to an opaque type like `type
    /// Foo<'a> = impl Trait<Item = &'a u32>`, but `impl Trait<item =
    /// dyn Bar>` expands to `type Foo = impl Trait<Item = dyn Bar +
    /// 'static>`. The latter uses `ImplicitObjectLifetimeDefault` so
    /// that surrounding code knows not to create a lifetime
    /// parameter.
    ImplicitObjectLifetimeDefault,

    /// Indicates an error during lowering (usually `'_` in wrong place)
    /// that was already reported.
    Error,

    /// User wrote specifies `'_`.
    Underscore,

    /// User wrote `'static`.
    Static,
}

impl LifetimeName {
    pub fn ident(&self) -> Ident {
        match *self {
            LifetimeName::ImplicitObjectLifetimeDefault
            | LifetimeName::Implicit
            | LifetimeName::Error => Ident::invalid(),
            LifetimeName::Underscore => Ident::with_dummy_span(kw::UnderscoreLifetime),
            LifetimeName::Static => Ident::with_dummy_span(kw::StaticLifetime),
            LifetimeName::Param(param_name) => param_name.ident(),
        }
    }

    pub fn is_elided(&self) -> bool {
        match self {
            LifetimeName::ImplicitObjectLifetimeDefault
            | LifetimeName::Implicit
            | LifetimeName::Underscore => true,

            // It might seem surprising that `Fresh(_)` counts as
            // *not* elided -- but this is because, as far as the code
            // in the compiler is concerned -- `Fresh(_)` variants act
            // equivalently to "some fresh name". They correspond to
            // early-bound regions on an impl, in other words.
            LifetimeName::Error | LifetimeName::Param(_) | LifetimeName::Static => false,
        }
    }

    fn is_static(&self) -> bool {
        self == &LifetimeName::Static
    }

    pub fn normalize_to_macros_2_0(&self) -> LifetimeName {
        match *self {
            LifetimeName::Param(param_name) => {
                LifetimeName::Param(param_name.normalize_to_macros_2_0())
            }
            lifetime_name => lifetime_name,
        }
    }
}

impl fmt::Display for Lifetime {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.name.ident().fmt(f)
    }
}

impl fmt::Debug for Lifetime {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "lifetime({}: {})", self.hir_id, self.name.ident())
    }
}

impl Lifetime {
    pub fn is_elided(&self) -> bool {
        self.name.is_elided()
    }

    pub fn is_static(&self) -> bool {
        self.name.is_static()
    }
}

/// A `Path` is essentially Rust's notion of a name; for instance,
/// `std::cmp::PartialEq`. It's represented as a sequence of identifiers,
/// along with a bunch of supporting information.
#[derive(Debug, HashStable_Generic)]
pub struct Path<'hir> {
    pub span: Span,
    /// The resolution for the path.
    pub res: Res,
    /// The segments in the path: the things separated by `::`.
    pub segments: &'hir [PathSegment<'hir>],
}

impl Path<'_> {
    pub fn is_global(&self) -> bool {
        !self.segments.is_empty() && self.segments[0].ident.name == kw::PathRoot
    }
}

/// A segment of a path: an identifier, an optional lifetime, and a set of
/// types.
#[derive(Debug, HashStable_Generic)]
pub struct PathSegment<'hir> {
    /// The identifier portion of this path segment.
    #[stable_hasher(project(name))]
    pub ident: Ident,
    // `id` and `res` are optional. We currently only use these in save-analysis,
    // any path segments without these will not have save-analysis info and
    // therefore will not have 'jump to def' in IDEs, but otherwise will not be
    // affected. (In general, we don't bother to get the defs for synthesized
    // segments, only for segments which have come from the AST).
    pub hir_id: Option<HirId>,
    pub res: Option<Res>,

    /// Type/lifetime parameters attached to this path. They come in
    /// two flavors: `Path<A,B,C>` and `Path(A,B) -> C`. Note that
    /// this is more than just simple syntactic sugar; the use of
    /// parens affects the region binding rules, so we preserve the
    /// distinction.
    pub args: Option<&'hir GenericArgs<'hir>>,

    /// Whether to infer remaining type parameters, if any.
    /// This only applies to expression and pattern paths, and
    /// out of those only the segments with no type parameters
    /// to begin with, e.g., `Vec::new` is `<Vec<..>>::new::<..>`.
    pub infer_args: bool,
}

impl<'hir> PathSegment<'hir> {
    /// Converts an identifier to the corresponding segment.
    pub fn from_ident(ident: Ident) -> PathSegment<'hir> {
        PathSegment { ident, hir_id: None, res: None, infer_args: true, args: None }
    }

    pub fn invalid() -> Self {
        Self::from_ident(Ident::invalid())
    }

    pub fn args(&self) -> &GenericArgs<'hir> {
        if let Some(ref args) = self.args {
            args
        } else {
            const DUMMY: &GenericArgs<'_> = &GenericArgs::none();
            DUMMY
        }
    }
}

#[derive(Encodable, Debug, HashStable_Generic)]
pub struct ConstArg {
    pub value: AnonConst,
    pub span: Span,
}

#[derive(Copy, Clone, Encodable, Debug, HashStable_Generic)]
pub enum InferKind {
    Const,
    Type,
}

impl InferKind {
    #[inline]
    pub fn is_type(self) -> bool {
        matches!(self, InferKind::Type)
    }
}

#[derive(Encodable, Debug, HashStable_Generic)]
pub struct InferArg {
    pub hir_id: HirId,
    pub kind: InferKind,
    pub span: Span,
}

impl InferArg {
    pub fn to_ty(&self) -> Ty<'_> {
        Ty { kind: TyKind::Infer, span: self.span, hir_id: self.hir_id }
    }
}

#[derive(Debug, HashStable_Generic)]
pub enum GenericArg<'hir> {
    Lifetime(Lifetime),
    Type(Ty<'hir>),
    Const(ConstArg),
    Infer(InferArg),
}

impl GenericArg<'_> {
    pub fn span(&self) -> Span {
        match self {
            GenericArg::Lifetime(l) => l.span,
            GenericArg::Type(t) => t.span,
            GenericArg::Const(c) => c.span,
            GenericArg::Infer(i) => i.span,
        }
    }

    pub fn id(&self) -> HirId {
        match self {
            GenericArg::Lifetime(l) => l.hir_id,
            GenericArg::Type(t) => t.hir_id,
            GenericArg::Const(c) => c.value.hir_id,
            GenericArg::Infer(i) => i.hir_id,
        }
    }

    pub fn is_const(&self) -> bool {
        matches!(self, GenericArg::Const(_))
    }

    pub fn is_synthetic(&self) -> bool {
        matches!(self, GenericArg::Lifetime(lifetime) if lifetime.name.ident() == Ident::invalid())
    }

    pub fn descr(&self) -> &'static str {
        match self {
            GenericArg::Lifetime(_) => "lifetime",
            GenericArg::Type(_) => "type",
            GenericArg::Const(_) => "constant",
            GenericArg::Infer(_) => "inferred",
        }
    }

    pub fn to_ord(&self, feats: &rustc_feature::Features) -> ast::ParamKindOrd {
        match self {
            GenericArg::Lifetime(_) => ast::ParamKindOrd::Lifetime,
            GenericArg::Type(_) => ast::ParamKindOrd::Type,
            GenericArg::Const(_) => {
                ast::ParamKindOrd::Const { unordered: feats.unordered_const_ty_params() }
            }
            GenericArg::Infer(_) => ast::ParamKindOrd::Infer,
        }
    }
}

#[derive(Debug, HashStable_Generic)]
pub struct GenericArgs<'hir> {
    /// The generic arguments for this path segment.
    pub args: &'hir [GenericArg<'hir>],
    /// Bindings (equality constraints) on associated types, if present.
    /// E.g., `Foo<A = Bar>`.
    pub bindings: &'hir [TypeBinding<'hir>],
    /// Were arguments written in parenthesized form `Fn(T) -> U`?
    /// This is required mostly for pretty-printing and diagnostics,
    /// but also for changing lifetime elision rules to be "function-like".
    pub parenthesized: bool,
    /// The span encompassing arguments and the surrounding brackets `<>` or `()`
    ///       Foo<A, B, AssocTy = D>           Fn(T, U, V) -> W
    ///          ^^^^^^^^^^^^^^^^^^^             ^^^^^^^^^
    /// Note that this may be:
    /// - empty, if there are no generic brackets (but there may be hidden lifetimes)
    /// - dummy, if this was generated while desugaring
    pub span_ext: Span,
}

impl GenericArgs<'_> {
    pub const fn none() -> Self {
        Self { args: &[], bindings: &[], parenthesized: false, span_ext: DUMMY_SP }
    }

    pub fn inputs(&self) -> &[Ty<'_>] {
        if self.parenthesized {
            for arg in self.args {
                match arg {
                    GenericArg::Lifetime(_) => {}
                    GenericArg::Type(ref ty) => {
                        if let TyKind::Tup(ref tys) = ty.kind {
                            return tys;
                        }
                        break;
                    }
                    GenericArg::Const(_) => {}
                    GenericArg::Infer(_) => {}
                }
            }
        }
        panic!("GenericArgs::inputs: not a `Fn(T) -> U`");
    }

    #[inline]
    pub fn has_type_params(&self) -> bool {
        self.args.iter().any(|arg| matches!(arg, GenericArg::Type(_)))
    }

    pub fn has_err(&self) -> bool {
        self.args.iter().any(|arg| match arg {
            GenericArg::Type(ty) => matches!(ty.kind, TyKind::Err),
            _ => false,
        }) || self.bindings.iter().any(|arg| match arg.kind {
            TypeBindingKind::Equality { ty } => matches!(ty.kind, TyKind::Err),
            _ => false,
        })
    }

    #[inline]
    pub fn num_type_params(&self) -> usize {
        self.args.iter().filter(|arg| matches!(arg, GenericArg::Type(_))).count()
    }

    #[inline]
    pub fn num_lifetime_params(&self) -> usize {
        self.args.iter().filter(|arg| matches!(arg, GenericArg::Lifetime(_))).count()
    }

    #[inline]
    pub fn has_lifetime_params(&self) -> bool {
        self.args.iter().any(|arg| matches!(arg, GenericArg::Lifetime(_)))
    }

    #[inline]
    pub fn num_generic_params(&self) -> usize {
        self.args.iter().filter(|arg| !matches!(arg, GenericArg::Lifetime(_))).count()
    }

    /// The span encompassing the text inside the surrounding brackets.
    /// It will also include bindings if they aren't in the form `-> Ret`
    /// Returns `None` if the span is empty (e.g. no brackets) or dummy
    pub fn span(&self) -> Option<Span> {
        let span_ext = self.span_ext()?;
        Some(span_ext.with_lo(span_ext.lo() + BytePos(1)).with_hi(span_ext.hi() - BytePos(1)))
    }

    /// Returns span encompassing arguments and their surrounding `<>` or `()`
    pub fn span_ext(&self) -> Option<Span> {
        Some(self.span_ext).filter(|span| !span.is_empty())
    }

    pub fn is_empty(&self) -> bool {
        self.args.is_empty()
    }
}

/// A modifier on a bound, currently this is only used for `?Sized`, where the
/// modifier is `Maybe`. Negative bounds should also be handled here.
#[derive(Copy, Clone, PartialEq, Eq, Encodable, Hash, Debug)]
#[derive(HashStable_Generic)]
pub enum TraitBoundModifier {
    None,
    Maybe,
    MaybeConst,
}

/// The AST represents all type param bounds as types.
/// `typeck::collect::compute_bounds` matches these against
/// the "special" built-in traits (see `middle::lang_items`) and
/// detects `Copy`, `Send` and `Sync`.
#[derive(Clone, Debug, HashStable_Generic)]
pub enum GenericBound<'hir> {
    Trait(PolyTraitRef<'hir>, TraitBoundModifier),
    // FIXME(davidtwco): Introduce `PolyTraitRef::LangItem`
    LangItemTrait(LangItem, Span, HirId, &'hir GenericArgs<'hir>),
    Outlives(Lifetime),
}

#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
rustc_data_structures::static_assert_size!(GenericBound<'_>, 48);

impl GenericBound<'_> {
    pub fn trait_ref(&self) -> Option<&TraitRef<'_>> {
        match self {
            GenericBound::Trait(data, _) => Some(&data.trait_ref),
            _ => None,
        }
    }

    pub fn span(&self) -> Span {
        match self {
            GenericBound::Trait(t, ..) => t.span,
            GenericBound::LangItemTrait(_, span, ..) => *span,
            GenericBound::Outlives(l) => l.span,
        }
    }
}

pub type GenericBounds<'hir> = &'hir [GenericBound<'hir>];

#[derive(Copy, Clone, PartialEq, Eq, Encodable, Debug, HashStable_Generic)]
pub enum LifetimeParamKind {
    // Indicates that the lifetime definition was explicitly declared (e.g., in
    // `fn foo<'a>(x: &'a u8) -> &'a u8 { x }`).
    Explicit,

    // Indicates that the lifetime definition was synthetically added
    // as a result of an in-band lifetime usage (e.g., in
    // `fn foo(x: &'a u8) -> &'a u8 { x }`).
    InBand,

    // Indication that the lifetime was elided (e.g., in both cases in
    // `fn foo(x: &u8) -> &'_ u8 { x }`).
    Elided,

    // Indication that the lifetime name was somehow in error.
    Error,
}

#[derive(Debug, HashStable_Generic)]
pub enum GenericParamKind<'hir> {
    /// A lifetime definition (e.g., `'a: 'b + 'c + 'd`).
    Lifetime {
        kind: LifetimeParamKind,
    },
    Type {
        default: Option<&'hir Ty<'hir>>,
        synthetic: Option<SyntheticTyParamKind>,
    },
    Const {
        ty: &'hir Ty<'hir>,
        /// Optional default value for the const generic param
        default: Option<AnonConst>,
    },
}

#[derive(Debug, HashStable_Generic)]
pub struct GenericParam<'hir> {
    pub hir_id: HirId,
    pub name: ParamName,
    pub bounds: GenericBounds<'hir>,
    pub span: Span,
    pub pure_wrt_drop: bool,
    pub kind: GenericParamKind<'hir>,
}

impl GenericParam<'hir> {
    pub fn bounds_span(&self) -> Option<Span> {
        self.bounds.iter().fold(None, |span, bound| {
            let span = span.map(|s| s.to(bound.span())).unwrap_or_else(|| bound.span());

            Some(span)
        })
    }
}

#[derive(Default)]
pub struct GenericParamCount {
    pub lifetimes: usize,
    pub types: usize,
    pub consts: usize,
    pub infer: usize,
}

/// Represents lifetimes and type parameters attached to a declaration
/// of a function, enum, trait, etc.
#[derive(Debug, HashStable_Generic)]
pub struct Generics<'hir> {
    pub params: &'hir [GenericParam<'hir>],
    pub where_clause: WhereClause<'hir>,
    pub span: Span,
}

impl Generics<'hir> {
    pub const fn empty() -> Generics<'hir> {
        Generics {
            params: &[],
            where_clause: WhereClause { predicates: &[], span: DUMMY_SP },
            span: DUMMY_SP,
        }
    }

    pub fn get_named(&self, name: Symbol) -> Option<&GenericParam<'_>> {
        for param in self.params {
            if name == param.name.ident().name {
                return Some(param);
            }
        }
        None
    }

    pub fn spans(&self) -> MultiSpan {
        if self.params.is_empty() {
            self.span.into()
        } else {
            self.params.iter().map(|p| p.span).collect::<Vec<Span>>().into()
        }
    }
}

/// Synthetic type parameters are converted to another form during lowering; this allows
/// us to track the original form they had, and is useful for error messages.
#[derive(Copy, Clone, PartialEq, Eq, Encodable, Decodable, Hash, Debug)]
#[derive(HashStable_Generic)]
pub enum SyntheticTyParamKind {
    ImplTrait,
    // Created by the `#[rustc_synthetic]` attribute.
    FromAttr,
}

/// A where-clause in a definition.
#[derive(Debug, HashStable_Generic)]
pub struct WhereClause<'hir> {
    pub predicates: &'hir [WherePredicate<'hir>],
    // Only valid if predicates aren't empty.
    pub span: Span,
}

impl WhereClause<'_> {
    pub fn span(&self) -> Option<Span> {
        if self.predicates.is_empty() { None } else { Some(self.span) }
    }

    /// The `WhereClause` under normal circumstances points at either the predicates or the empty
    /// space where the `where` clause should be. Only of use for diagnostic suggestions.
    pub fn span_for_predicates_or_empty_place(&self) -> Span {
        self.span
    }

    /// `Span` where further predicates would be suggested, accounting for trailing commas, like
    ///  in `fn foo<T>(t: T) where T: Foo,` so we don't suggest two trailing commas.
    pub fn tail_span_for_suggestion(&self) -> Span {
        let end = self.span_for_predicates_or_empty_place().shrink_to_hi();
        self.predicates.last().map_or(end, |p| p.span()).shrink_to_hi().to(end)
    }
}

/// A single predicate in a where-clause.
#[derive(Debug, HashStable_Generic)]
pub enum WherePredicate<'hir> {
    /// A type binding (e.g., `for<'c> Foo: Send + Clone + 'c`).
    BoundPredicate(WhereBoundPredicate<'hir>),
    /// A lifetime predicate (e.g., `'a: 'b + 'c`).
    RegionPredicate(WhereRegionPredicate<'hir>),
    /// An equality predicate (unsupported).
    EqPredicate(WhereEqPredicate<'hir>),
}

impl WherePredicate<'_> {
    pub fn span(&self) -> Span {
        match self {
            WherePredicate::BoundPredicate(p) => p.span,
            WherePredicate::RegionPredicate(p) => p.span,
            WherePredicate::EqPredicate(p) => p.span,
        }
    }
}

/// A type bound (e.g., `for<'c> Foo: Send + Clone + 'c`).
#[derive(Debug, HashStable_Generic)]
pub struct WhereBoundPredicate<'hir> {
    pub span: Span,
    /// Any generics from a `for` binding.
    pub bound_generic_params: &'hir [GenericParam<'hir>],
    /// The type being bounded.
    pub bounded_ty: &'hir Ty<'hir>,
    /// Trait and lifetime bounds (e.g., `Clone + Send + 'static`).
    pub bounds: GenericBounds<'hir>,
}

/// A lifetime predicate (e.g., `'a: 'b + 'c`).
#[derive(Debug, HashStable_Generic)]
pub struct WhereRegionPredicate<'hir> {
    pub span: Span,
    pub lifetime: Lifetime,
    pub bounds: GenericBounds<'hir>,
}

/// An equality predicate (e.g., `T = int`); currently unsupported.
#[derive(Debug, HashStable_Generic)]
pub struct WhereEqPredicate<'hir> {
    pub hir_id: HirId,
    pub span: Span,
    pub lhs_ty: &'hir Ty<'hir>,
    pub rhs_ty: &'hir Ty<'hir>,
}

/// The top-level data structure that stores the entire contents of
/// the crate currently being compiled.
///
/// For more details, see the [rustc dev guide].
///
/// [rustc dev guide]: https://rustc-dev-guide.rust-lang.org/hir.html
#[derive(Debug)]
pub struct Crate<'hir> {
    pub owners: IndexVec<LocalDefId, Option<OwnerNode<'hir>>>,
    pub bodies: BTreeMap<BodyId, Body<'hir>>,

    /// Map indicating what traits are in scope for places where this
    /// is relevant; generated by resolve.
    pub trait_map: FxHashMap<LocalDefId, FxHashMap<ItemLocalId, Box<[TraitCandidate]>>>,

    /// Collected attributes from HIR nodes.
    pub attrs: BTreeMap<HirId, &'hir [Attribute]>,
}

impl Crate<'hir> {
    pub fn module(&self) -> &'hir Mod<'hir> {
        if let Some(OwnerNode::Crate(m)) = self.owners[CRATE_DEF_ID] { m } else { panic!() }
    }

    pub fn item(&self, id: ItemId) -> &'hir Item<'hir> {
        self.owners[id.def_id].as_ref().unwrap().expect_item()
    }

    pub fn trait_item(&self, id: TraitItemId) -> &'hir TraitItem<'hir> {
        self.owners[id.def_id].as_ref().unwrap().expect_trait_item()
    }

    pub fn impl_item(&self, id: ImplItemId) -> &'hir ImplItem<'hir> {
        self.owners[id.def_id].as_ref().unwrap().expect_impl_item()
    }

    pub fn foreign_item(&self, id: ForeignItemId) -> &'hir ForeignItem<'hir> {
        self.owners[id.def_id].as_ref().unwrap().expect_foreign_item()
    }

    pub fn body(&self, id: BodyId) -> &Body<'hir> {
        &self.bodies[&id]
    }
}

/// A block of statements `{ .. }`, which may have a label (in this case the
/// `targeted_by_break` field will be `true`) and may be `unsafe` by means of
/// the `rules` being anything but `DefaultBlock`.
#[derive(Debug, HashStable_Generic)]
pub struct Block<'hir> {
    /// Statements in a block.
    pub stmts: &'hir [Stmt<'hir>],
    /// An expression at the end of the block
    /// without a semicolon, if any.
    pub expr: Option<&'hir Expr<'hir>>,
    #[stable_hasher(ignore)]
    pub hir_id: HirId,
    /// Distinguishes between `unsafe { ... }` and `{ ... }`.
    pub rules: BlockCheckMode,
    pub span: Span,
    /// If true, then there may exist `break 'a` values that aim to
    /// break out of this block early.
    /// Used by `'label: {}` blocks and by `try {}` blocks.
    pub targeted_by_break: bool,
}

#[derive(Debug, HashStable_Generic)]
pub struct Pat<'hir> {
    #[stable_hasher(ignore)]
    pub hir_id: HirId,
    pub kind: PatKind<'hir>,
    pub span: Span,
    // Whether to use default binding modes.
    // At present, this is false only for destructuring assignment.
    pub default_binding_modes: bool,
}

impl<'hir> Pat<'hir> {
    // FIXME(#19596) this is a workaround, but there should be a better way
    fn walk_short_(&self, it: &mut impl FnMut(&Pat<'hir>) -> bool) -> bool {
        if !it(self) {
            return false;
        }

        use PatKind::*;
        match self.kind {
            Wild | Lit(_) | Range(..) | Binding(.., None) | Path(_) => true,
            Box(s) | Ref(s, _) | Binding(.., Some(s)) => s.walk_short_(it),
            Struct(_, fields, _) => fields.iter().all(|field| field.pat.walk_short_(it)),
            TupleStruct(_, s, _) | Tuple(s, _) | Or(s) => s.iter().all(|p| p.walk_short_(it)),
            Slice(before, slice, after) => {
                before.iter().chain(slice).chain(after.iter()).all(|p| p.walk_short_(it))
            }
        }
    }

    /// Walk the pattern in left-to-right order,
    /// short circuiting (with `.all(..)`) if `false` is returned.
    ///
    /// Note that when visiting e.g. `Tuple(ps)`,
    /// if visiting `ps[0]` returns `false`,
    /// then `ps[1]` will not be visited.
    pub fn walk_short(&self, mut it: impl FnMut(&Pat<'hir>) -> bool) -> bool {
        self.walk_short_(&mut it)
    }

    // FIXME(#19596) this is a workaround, but there should be a better way
    fn walk_(&self, it: &mut impl FnMut(&Pat<'hir>) -> bool) {
        if !it(self) {
            return;
        }

        use PatKind::*;
        match self.kind {
            Wild | Lit(_) | Range(..) | Binding(.., None) | Path(_) => {}
            Box(s) | Ref(s, _) | Binding(.., Some(s)) => s.walk_(it),
            Struct(_, fields, _) => fields.iter().for_each(|field| field.pat.walk_(it)),
            TupleStruct(_, s, _) | Tuple(s, _) | Or(s) => s.iter().for_each(|p| p.walk_(it)),
            Slice(before, slice, after) => {
                before.iter().chain(slice).chain(after.iter()).for_each(|p| p.walk_(it))
            }
        }
    }

    /// Walk the pattern in left-to-right order.
    ///
    /// If `it(pat)` returns `false`, the children are not visited.
    pub fn walk(&self, mut it: impl FnMut(&Pat<'hir>) -> bool) {
        self.walk_(&mut it)
    }

    /// Walk the pattern in left-to-right order.
    ///
    /// If you always want to recurse, prefer this method over `walk`.
    pub fn walk_always(&self, mut it: impl FnMut(&Pat<'_>)) {
        self.walk(|p| {
            it(p);
            true
        })
    }
}

/// A single field in a struct pattern.
///
/// Patterns like the fields of Foo `{ x, ref y, ref mut z }`
/// are treated the same as` x: x, y: ref y, z: ref mut z`,
/// except `is_shorthand` is true.
#[derive(Debug, HashStable_Generic)]
pub struct PatField<'hir> {
    #[stable_hasher(ignore)]
    pub hir_id: HirId,
    /// The identifier for the field.
    #[stable_hasher(project(name))]
    pub ident: Ident,
    /// The pattern the field is destructured to.
    pub pat: &'hir Pat<'hir>,
    pub is_shorthand: bool,
    pub span: Span,
}

/// Explicit binding annotations given in the HIR for a binding. Note
/// that this is not the final binding *mode* that we infer after type
/// inference.
#[derive(Copy, Clone, PartialEq, Encodable, Debug, HashStable_Generic)]
pub enum BindingAnnotation {
    /// No binding annotation given: this means that the final binding mode
    /// will depend on whether we have skipped through a `&` reference
    /// when matching. For example, the `x` in `Some(x)` will have binding
    /// mode `None`; if you do `let Some(x) = &Some(22)`, it will
    /// ultimately be inferred to be by-reference.
    ///
    /// Note that implicit reference skipping is not implemented yet (#42640).
    Unannotated,

    /// Annotated with `mut x` -- could be either ref or not, similar to `None`.
    Mutable,

    /// Annotated as `ref`, like `ref x`
    Ref,

    /// Annotated as `ref mut x`.
    RefMut,
}

#[derive(Copy, Clone, PartialEq, Encodable, Debug, HashStable_Generic)]
pub enum RangeEnd {
    Included,
    Excluded,
}

impl fmt::Display for RangeEnd {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            RangeEnd::Included => "..=",
            RangeEnd::Excluded => "..",
        })
    }
}

#[derive(Debug, HashStable_Generic)]
pub enum PatKind<'hir> {
    /// Represents a wildcard pattern (i.e., `_`).
    Wild,

    /// A fresh binding `ref mut binding @ OPT_SUBPATTERN`.
    /// The `HirId` is the canonical ID for the variable being bound,
    /// (e.g., in `Ok(x) | Err(x)`, both `x` use the same canonical ID),
    /// which is the pattern ID of the first `x`.
    Binding(BindingAnnotation, HirId, Ident, Option<&'hir Pat<'hir>>),

    /// A struct or struct variant pattern (e.g., `Variant {x, y, ..}`).
    /// The `bool` is `true` in the presence of a `..`.
    Struct(QPath<'hir>, &'hir [PatField<'hir>], bool),

    /// A tuple struct/variant pattern `Variant(x, y, .., z)`.
    /// If the `..` pattern fragment is present, then `Option<usize>` denotes its position.
    /// `0 <= position <= subpats.len()`
    TupleStruct(QPath<'hir>, &'hir [Pat<'hir>], Option<usize>),

    /// An or-pattern `A | B | C`.
    /// Invariant: `pats.len() >= 2`.
    Or(&'hir [Pat<'hir>]),

    /// A path pattern for a unit struct/variant or a (maybe-associated) constant.
    Path(QPath<'hir>),

    /// A tuple pattern (e.g., `(a, b)`).
    /// If the `..` pattern fragment is present, then `Option<usize>` denotes its position.
    /// `0 <= position <= subpats.len()`
    Tuple(&'hir [Pat<'hir>], Option<usize>),

    /// A `box` pattern.
    Box(&'hir Pat<'hir>),

    /// A reference pattern (e.g., `&mut (a, b)`).
    Ref(&'hir Pat<'hir>, Mutability),

    /// A literal.
    Lit(&'hir Expr<'hir>),

    /// A range pattern (e.g., `1..=2` or `1..2`).
    Range(Option<&'hir Expr<'hir>>, Option<&'hir Expr<'hir>>, RangeEnd),

    /// A slice pattern, `[before_0, ..., before_n, (slice, after_0, ..., after_n)?]`.
    ///
    /// Here, `slice` is lowered from the syntax `($binding_mode $ident @)? ..`.
    /// If `slice` exists, then `after` can be non-empty.
    ///
    /// The representation for e.g., `[a, b, .., c, d]` is:
    /// ```
    /// PatKind::Slice([Binding(a), Binding(b)], Some(Wild), [Binding(c), Binding(d)])
    /// ```
    Slice(&'hir [Pat<'hir>], Option<&'hir Pat<'hir>>, &'hir [Pat<'hir>]),
}

#[derive(Copy, Clone, PartialEq, Encodable, Debug, HashStable_Generic)]
pub enum BinOpKind {
    /// The `+` operator (addition).
    Add,
    /// The `-` operator (subtraction).
    Sub,
    /// The `*` operator (multiplication).
    Mul,
    /// The `/` operator (division).
    Div,
    /// The `%` operator (modulus).
    Rem,
    /// The `&&` operator (logical and).
    And,
    /// The `||` operator (logical or).
    Or,
    /// The `^` operator (bitwise xor).
    BitXor,
    /// The `&` operator (bitwise and).
    BitAnd,
    /// The `|` operator (bitwise or).
    BitOr,
    /// The `<<` operator (shift left).
    Shl,
    /// The `>>` operator (shift right).
    Shr,
    /// The `==` operator (equality).
    Eq,
    /// The `<` operator (less than).
    Lt,
    /// The `<=` operator (less than or equal to).
    Le,
    /// The `!=` operator (not equal to).
    Ne,
    /// The `>=` operator (greater than or equal to).
    Ge,
    /// The `>` operator (greater than).
    Gt,
}

impl BinOpKind {
    pub fn as_str(self) -> &'static str {
        match self {
            BinOpKind::Add => "+",
            BinOpKind::Sub => "-",
            BinOpKind::Mul => "*",
            BinOpKind::Div => "/",
            BinOpKind::Rem => "%",
            BinOpKind::And => "&&",
            BinOpKind::Or => "||",
            BinOpKind::BitXor => "^",
            BinOpKind::BitAnd => "&",
            BinOpKind::BitOr => "|",
            BinOpKind::Shl => "<<",
            BinOpKind::Shr => ">>",
            BinOpKind::Eq => "==",
            BinOpKind::Lt => "<",
            BinOpKind::Le => "<=",
            BinOpKind::Ne => "!=",
            BinOpKind::Ge => ">=",
            BinOpKind::Gt => ">",
        }
    }

    pub fn is_lazy(self) -> bool {
        matches!(self, BinOpKind::And | BinOpKind::Or)
    }

    pub fn is_shift(self) -> bool {
        matches!(self, BinOpKind::Shl | BinOpKind::Shr)
    }

    pub fn is_comparison(self) -> bool {
        match self {
            BinOpKind::Eq
            | BinOpKind::Lt
            | BinOpKind::Le
            | BinOpKind::Ne
            | BinOpKind::Gt
            | BinOpKind::Ge => true,
            BinOpKind::And
            | BinOpKind::Or
            | BinOpKind::Add
            | BinOpKind::Sub
            | BinOpKind::Mul
            | BinOpKind::Div
            | BinOpKind::Rem
            | BinOpKind::BitXor
            | BinOpKind::BitAnd
            | BinOpKind::BitOr
            | BinOpKind::Shl
            | BinOpKind::Shr => false,
        }
    }

    /// Returns `true` if the binary operator takes its arguments by value.
    pub fn is_by_value(self) -> bool {
        !self.is_comparison()
    }
}

impl Into<ast::BinOpKind> for BinOpKind {
    fn into(self) -> ast::BinOpKind {
        match self {
            BinOpKind::Add => ast::BinOpKind::Add,
            BinOpKind::Sub => ast::BinOpKind::Sub,
            BinOpKind::Mul => ast::BinOpKind::Mul,
            BinOpKind::Div => ast::BinOpKind::Div,
            BinOpKind::Rem => ast::BinOpKind::Rem,
            BinOpKind::And => ast::BinOpKind::And,
            BinOpKind::Or => ast::BinOpKind::Or,
            BinOpKind::BitXor => ast::BinOpKind::BitXor,
            BinOpKind::BitAnd => ast::BinOpKind::BitAnd,
            BinOpKind::BitOr => ast::BinOpKind::BitOr,
            BinOpKind::Shl => ast::BinOpKind::Shl,
            BinOpKind::Shr => ast::BinOpKind::Shr,
            BinOpKind::Eq => ast::BinOpKind::Eq,
            BinOpKind::Lt => ast::BinOpKind::Lt,
            BinOpKind::Le => ast::BinOpKind::Le,
            BinOpKind::Ne => ast::BinOpKind::Ne,
            BinOpKind::Ge => ast::BinOpKind::Ge,
            BinOpKind::Gt => ast::BinOpKind::Gt,
        }
    }
}

pub type BinOp = Spanned<BinOpKind>;

#[derive(Copy, Clone, PartialEq, Encodable, Debug, HashStable_Generic)]
pub enum UnOp {
    /// The `*` operator (deferencing).
    Deref,
    /// The `!` operator (logical negation).
    Not,
    /// The `-` operator (negation).
    Neg,
}

impl UnOp {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Deref => "*",
            Self::Not => "!",
            Self::Neg => "-",
        }
    }

    /// Returns `true` if the unary operator takes its argument by value.
    pub fn is_by_value(self) -> bool {
        matches!(self, Self::Neg | Self::Not)
    }
}

/// A statement.
#[derive(Debug, HashStable_Generic)]
pub struct Stmt<'hir> {
    pub hir_id: HirId,
    pub kind: StmtKind<'hir>,
    pub span: Span,
}

/// The contents of a statement.
#[derive(Debug, HashStable_Generic)]
pub enum StmtKind<'hir> {
    /// A local (`let`) binding.
    Local(&'hir Local<'hir>),

    /// An item binding.
    Item(ItemId),

    /// An expression without a trailing semi-colon (must have unit type).
    Expr(&'hir Expr<'hir>),

    /// An expression with a trailing semi-colon (may have any type).
    Semi(&'hir Expr<'hir>),
}

/// Represents a `let` statement (i.e., `let <pat>:<ty> = <expr>;`).
#[derive(Debug, HashStable_Generic)]
pub struct Local<'hir> {
    pub pat: &'hir Pat<'hir>,
    /// Type annotation, if any (otherwise the type will be inferred).
    pub ty: Option<&'hir Ty<'hir>>,
    /// Initializer expression to set the value, if any.
    pub init: Option<&'hir Expr<'hir>>,
    pub hir_id: HirId,
    pub span: Span,
    /// Can be `ForLoopDesugar` if the `let` statement is part of a `for` loop
    /// desugaring. Otherwise will be `Normal`.
    pub source: LocalSource,
}

/// Represents a single arm of a `match` expression, e.g.
/// `<pat> (if <guard>) => <body>`.
#[derive(Debug, HashStable_Generic)]
pub struct Arm<'hir> {
    #[stable_hasher(ignore)]
    pub hir_id: HirId,
    pub span: Span,
    /// If this pattern and the optional guard matches, then `body` is evaluated.
    pub pat: &'hir Pat<'hir>,
    /// Optional guard clause.
    pub guard: Option<Guard<'hir>>,
    /// The expression the arm evaluates to if this arm matches.
    pub body: &'hir Expr<'hir>,
}

#[derive(Debug, HashStable_Generic)]
pub enum Guard<'hir> {
    If(&'hir Expr<'hir>),
    // FIXME use ExprKind::Let for this.
    IfLet(&'hir Pat<'hir>, &'hir Expr<'hir>),
}

#[derive(Debug, HashStable_Generic)]
pub struct ExprField<'hir> {
    #[stable_hasher(ignore)]
    pub hir_id: HirId,
    pub ident: Ident,
    pub expr: &'hir Expr<'hir>,
    pub span: Span,
    pub is_shorthand: bool,
}

#[derive(Copy, Clone, PartialEq, Encodable, Debug, HashStable_Generic)]
pub enum BlockCheckMode {
    DefaultBlock,
    UnsafeBlock(UnsafeSource),
}

#[derive(Copy, Clone, PartialEq, Encodable, Debug, HashStable_Generic)]
pub enum UnsafeSource {
    CompilerGenerated,
    UserProvided,
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Encodable, Hash, Debug)]
pub struct BodyId {
    pub hir_id: HirId,
}

/// The body of a function, closure, or constant value. In the case of
/// a function, the body contains not only the function body itself
/// (which is an expression), but also the argument patterns, since
/// those are something that the caller doesn't really care about.
///
/// # Examples
///
/// ```
/// fn foo((x, y): (u32, u32)) -> u32 {
///     x + y
/// }
/// ```
///
/// Here, the `Body` associated with `foo()` would contain:
///
/// - an `params` array containing the `(x, y)` pattern
/// - a `value` containing the `x + y` expression (maybe wrapped in a block)
/// - `generator_kind` would be `None`
///
/// All bodies have an **owner**, which can be accessed via the HIR
/// map using `body_owner_def_id()`.
#[derive(Debug)]
pub struct Body<'hir> {
    pub params: &'hir [Param<'hir>],
    pub value: Expr<'hir>,
    pub generator_kind: Option<GeneratorKind>,
}

impl Body<'hir> {
    pub fn id(&self) -> BodyId {
        BodyId { hir_id: self.value.hir_id }
    }

    pub fn generator_kind(&self) -> Option<GeneratorKind> {
        self.generator_kind
    }
}

/// The type of source expression that caused this generator to be created.
#[derive(
    Clone,
    PartialEq,
    PartialOrd,
    Eq,
    Hash,
    HashStable_Generic,
    Encodable,
    Decodable,
    Debug,
    Copy
)]
pub enum GeneratorKind {
    /// An explicit `async` block or the body of an async function.
    Async(AsyncGeneratorKind),

    /// A generator literal created via a `yield` inside a closure.
    Gen,
}

impl fmt::Display for GeneratorKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GeneratorKind::Async(k) => fmt::Display::fmt(k, f),
            GeneratorKind::Gen => f.write_str("generator"),
        }
    }
}

impl GeneratorKind {
    pub fn descr(&self) -> &'static str {
        match self {
            GeneratorKind::Async(ask) => ask.descr(),
            GeneratorKind::Gen => "generator",
        }
    }
}

/// In the case of a generator created as part of an async construct,
/// which kind of async construct caused it to be created?
///
/// This helps error messages but is also used to drive coercions in
/// type-checking (see #60424).
#[derive(
    Clone,
    PartialEq,
    PartialOrd,
    Eq,
    Hash,
    HashStable_Generic,
    Encodable,
    Decodable,
    Debug,
    Copy
)]
pub enum AsyncGeneratorKind {
    /// An explicit `async` block written by the user.
    Block,

    /// An explicit `async` block written by the user.
    Closure,

    /// The `async` block generated as the body of an async function.
    Fn,
}

impl fmt::Display for AsyncGeneratorKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            AsyncGeneratorKind::Block => "`async` block",
            AsyncGeneratorKind::Closure => "`async` closure body",
            AsyncGeneratorKind::Fn => "`async fn` body",
        })
    }
}

impl AsyncGeneratorKind {
    pub fn descr(&self) -> &'static str {
        match self {
            AsyncGeneratorKind::Block => "`async` block",
            AsyncGeneratorKind::Closure => "`async` closure body",
            AsyncGeneratorKind::Fn => "`async fn` body",
        }
    }
}

#[derive(Copy, Clone, Debug)]
pub enum BodyOwnerKind {
    /// Functions and methods.
    Fn,

    /// Closures
    Closure,

    /// Constants and associated constants.
    Const,

    /// Initializer of a `static` item.
    Static(Mutability),
}

impl BodyOwnerKind {
    pub fn is_fn_or_closure(self) -> bool {
        match self {
            BodyOwnerKind::Fn | BodyOwnerKind::Closure => true,
            BodyOwnerKind::Const | BodyOwnerKind::Static(_) => false,
        }
    }
}

/// The kind of an item that requires const-checking.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ConstContext {
    /// A `const fn`.
    ConstFn,

    /// A `static` or `static mut`.
    Static(Mutability),

    /// A `const`, associated `const`, or other const context.
    ///
    /// Other contexts include:
    /// - Array length expressions
    /// - Enum discriminants
    /// - Const generics
    ///
    /// For the most part, other contexts are treated just like a regular `const`, so they are
    /// lumped into the same category.
    Const,
}

impl ConstContext {
    /// A description of this const context that can appear between backticks in an error message.
    ///
    /// E.g. `const` or `static mut`.
    pub fn keyword_name(self) -> &'static str {
        match self {
            Self::Const => "const",
            Self::Static(Mutability::Not) => "static",
            Self::Static(Mutability::Mut) => "static mut",
            Self::ConstFn => "const fn",
        }
    }
}

/// A colloquial, trivially pluralizable description of this const context for use in error
/// messages.
impl fmt::Display for ConstContext {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Self::Const => write!(f, "constant"),
            Self::Static(_) => write!(f, "static"),
            Self::ConstFn => write!(f, "constant function"),
        }
    }
}

/// A literal.
pub type Lit = Spanned<LitKind>;

/// A constant (expression) that's not an item or associated item,
/// but needs its own `DefId` for type-checking, const-eval, etc.
/// These are usually found nested inside types (e.g., array lengths)
/// or expressions (e.g., repeat counts), and also used to define
/// explicit discriminant values for enum variants.
///
/// You can check if this anon const is a default in a const param
/// `const N: usize = { ... }` with `tcx.hir().opt_const_param_default_param_hir_id(..)`
#[derive(Copy, Clone, PartialEq, Eq, Encodable, Debug, HashStable_Generic)]
pub struct AnonConst {
    pub hir_id: HirId,
    pub body: BodyId,
}

/// An expression.
#[derive(Debug)]
pub struct Expr<'hir> {
    pub hir_id: HirId,
    pub kind: ExprKind<'hir>,
    pub span: Span,
}

impl Expr<'_> {
    pub fn precedence(&self) -> ExprPrecedence {
        match self.kind {
            ExprKind::Box(_) => ExprPrecedence::Box,
            ExprKind::ConstBlock(_) => ExprPrecedence::ConstBlock,
            ExprKind::Array(_) => ExprPrecedence::Array,
            ExprKind::Call(..) => ExprPrecedence::Call,
            ExprKind::MethodCall(..) => ExprPrecedence::MethodCall,
            ExprKind::Tup(_) => ExprPrecedence::Tup,
            ExprKind::Binary(op, ..) => ExprPrecedence::Binary(op.node.into()),
            ExprKind::Unary(..) => ExprPrecedence::Unary,
            ExprKind::Lit(_) => ExprPrecedence::Lit,
            ExprKind::Type(..) | ExprKind::Cast(..) => ExprPrecedence::Cast,
            ExprKind::DropTemps(ref expr, ..) => expr.precedence(),
            ExprKind::If(..) => ExprPrecedence::If,
            ExprKind::Let(..) => ExprPrecedence::Let,
            ExprKind::Loop(..) => ExprPrecedence::Loop,
            ExprKind::Match(..) => ExprPrecedence::Match,
            ExprKind::Closure(..) => ExprPrecedence::Closure,
            ExprKind::Block(..) => ExprPrecedence::Block,
            ExprKind::Assign(..) => ExprPrecedence::Assign,
            ExprKind::AssignOp(..) => ExprPrecedence::AssignOp,
            ExprKind::Field(..) => ExprPrecedence::Field,
            ExprKind::Index(..) => ExprPrecedence::Index,
            ExprKind::Path(..) => ExprPrecedence::Path,
            ExprKind::AddrOf(..) => ExprPrecedence::AddrOf,
            ExprKind::Break(..) => ExprPrecedence::Break,
            ExprKind::Continue(..) => ExprPrecedence::Continue,
            ExprKind::Ret(..) => ExprPrecedence::Ret,
            ExprKind::InlineAsm(..) => ExprPrecedence::InlineAsm,
            ExprKind::LlvmInlineAsm(..) => ExprPrecedence::InlineAsm,
            ExprKind::Struct(..) => ExprPrecedence::Struct,
            ExprKind::Repeat(..) => ExprPrecedence::Repeat,
            ExprKind::Yield(..) => ExprPrecedence::Yield,
            ExprKind::Err => ExprPrecedence::Err,
        }
    }

    // Whether this looks like a place expr, without checking for deref
    // adjustments.
    // This will return `true` in some potentially surprising cases such as
    // `CONSTANT.field`.
    pub fn is_syntactic_place_expr(&self) -> bool {
        self.is_place_expr(|_| true)
    }

    /// Whether this is a place expression.
    ///
    /// `allow_projections_from` should return `true` if indexing a field or index expression based
    /// on the given expression should be considered a place expression.
    pub fn is_place_expr(&self, mut allow_projections_from: impl FnMut(&Self) -> bool) -> bool {
        match self.kind {
            ExprKind::Path(QPath::Resolved(_, ref path)) => {
                matches!(path.res, Res::Local(..) | Res::Def(DefKind::Static, _) | Res::Err)
            }

            // Type ascription inherits its place expression kind from its
            // operand. See:
            // https://github.com/rust-lang/rfcs/blob/master/text/0803-type-ascription.md#type-ascription-and-temporaries
            ExprKind::Type(ref e, _) => e.is_place_expr(allow_projections_from),

            ExprKind::Unary(UnOp::Deref, _) => true,

            ExprKind::Field(ref base, _) | ExprKind::Index(ref base, _) => {
                allow_projections_from(base) || base.is_place_expr(allow_projections_from)
            }

            // Lang item paths cannot currently be local variables or statics.
            ExprKind::Path(QPath::LangItem(..)) => false,

            // Partially qualified paths in expressions can only legally
            // refer to associated items which are always rvalues.
            ExprKind::Path(QPath::TypeRelative(..))
            | ExprKind::Call(..)
            | ExprKind::MethodCall(..)
            | ExprKind::Struct(..)
            | ExprKind::Tup(..)
            | ExprKind::If(..)
            | ExprKind::Match(..)
            | ExprKind::Closure(..)
            | ExprKind::Block(..)
            | ExprKind::Repeat(..)
            | ExprKind::Array(..)
            | ExprKind::Break(..)
            | ExprKind::Continue(..)
            | ExprKind::Ret(..)
            | ExprKind::Let(..)
            | ExprKind::Loop(..)
            | ExprKind::Assign(..)
            | ExprKind::InlineAsm(..)
            | ExprKind::LlvmInlineAsm(..)
            | ExprKind::AssignOp(..)
            | ExprKind::Lit(_)
            | ExprKind::ConstBlock(..)
            | ExprKind::Unary(..)
            | ExprKind::Box(..)
            | ExprKind::AddrOf(..)
            | ExprKind::Binary(..)
            | ExprKind::Yield(..)
            | ExprKind::Cast(..)
            | ExprKind::DropTemps(..)
            | ExprKind::Err => false,
        }
    }

    /// If `Self.kind` is `ExprKind::DropTemps(expr)`, drill down until we get a non-`DropTemps`
    /// `Expr`. This is used in suggestions to ignore this `ExprKind` as it is semantically
    /// silent, only signaling the ownership system. By doing this, suggestions that check the
    /// `ExprKind` of any given `Expr` for presentation don't have to care about `DropTemps`
    /// beyond remembering to call this function before doing analysis on it.
    pub fn peel_drop_temps(&self) -> &Self {
        let mut expr = self;
        while let ExprKind::DropTemps(inner) = &expr.kind {
            expr = inner;
        }
        expr
    }

    pub fn peel_blocks(&self) -> &Self {
        let mut expr = self;
        while let ExprKind::Block(Block { expr: Some(inner), .. }, _) = &expr.kind {
            expr = inner;
        }
        expr
    }

    pub fn can_have_side_effects(&self) -> bool {
        match self.peel_drop_temps().kind {
            ExprKind::Path(_) | ExprKind::Lit(_) => false,
            ExprKind::Type(base, _)
            | ExprKind::Unary(_, base)
            | ExprKind::Field(base, _)
            | ExprKind::Index(base, _)
            | ExprKind::AddrOf(.., base)
            | ExprKind::Cast(base, _) => {
                // This isn't exactly true for `Index` and all `Unnary`, but we are using this
                // method exclusively for diagnostics and there's a *cultural* pressure against
                // them being used only for its side-effects.
                base.can_have_side_effects()
            }
            ExprKind::Struct(_, fields, init) => fields
                .iter()
                .map(|field| field.expr)
                .chain(init.into_iter())
                .all(|e| e.can_have_side_effects()),

            ExprKind::Array(args)
            | ExprKind::Tup(args)
            | ExprKind::Call(
                Expr {
                    kind:
                        ExprKind::Path(QPath::Resolved(
                            None,
                            Path { res: Res::Def(DefKind::Ctor(_, CtorKind::Fn), _), .. },
                        )),
                    ..
                },
                args,
            ) => args.iter().all(|arg| arg.can_have_side_effects()),
            ExprKind::If(..)
            | ExprKind::Match(..)
            | ExprKind::MethodCall(..)
            | ExprKind::Call(..)
            | ExprKind::Closure(..)
            | ExprKind::Block(..)
            | ExprKind::Repeat(..)
            | ExprKind::Break(..)
            | ExprKind::Continue(..)
            | ExprKind::Ret(..)
            | ExprKind::Let(..)
            | ExprKind::Loop(..)
            | ExprKind::Assign(..)
            | ExprKind::InlineAsm(..)
            | ExprKind::LlvmInlineAsm(..)
            | ExprKind::AssignOp(..)
            | ExprKind::ConstBlock(..)
            | ExprKind::Box(..)
            | ExprKind::Binary(..)
            | ExprKind::Yield(..)
            | ExprKind::DropTemps(..)
            | ExprKind::Err => true,
        }
    }
}

/// Checks if the specified expression is a built-in range literal.
/// (See: `LoweringContext::lower_expr()`).
pub fn is_range_literal(expr: &Expr<'_>) -> bool {
    match expr.kind {
        // All built-in range literals but `..=` and `..` desugar to `Struct`s.
        ExprKind::Struct(ref qpath, _, _) => matches!(
            **qpath,
            QPath::LangItem(
                LangItem::Range
                    | LangItem::RangeTo
                    | LangItem::RangeFrom
                    | LangItem::RangeFull
                    | LangItem::RangeToInclusive,
                _,
            )
        ),

        // `..=` desugars into `::std::ops::RangeInclusive::new(...)`.
        ExprKind::Call(ref func, _) => {
            matches!(func.kind, ExprKind::Path(QPath::LangItem(LangItem::RangeInclusiveNew, _)))
        }

        _ => false,
    }
}

#[derive(Debug, HashStable_Generic)]
pub enum ExprKind<'hir> {
    /// A `box x` expression.
    Box(&'hir Expr<'hir>),
    /// Allow anonymous constants from an inline `const` block
    ConstBlock(AnonConst),
    /// An array (e.g., `[a, b, c, d]`).
    Array(&'hir [Expr<'hir>]),
    /// A function call.
    ///
    /// The first field resolves to the function itself (usually an `ExprKind::Path`),
    /// and the second field is the list of arguments.
    /// This also represents calling the constructor of
    /// tuple-like ADTs such as tuple structs and enum variants.
    Call(&'hir Expr<'hir>, &'hir [Expr<'hir>]),
    /// A method call (e.g., `x.foo::<'static, Bar, Baz>(a, b, c, d)`).
    ///
    /// The `PathSegment`/`Span` represent the method name and its generic arguments
    /// (within the angle brackets).
    /// The first element of the vector of `Expr`s is the expression that evaluates
    /// to the object on which the method is being called on (the receiver),
    /// and the remaining elements are the rest of the arguments.
    /// Thus, `x.foo::<Bar, Baz>(a, b, c, d)` is represented as
    /// `ExprKind::MethodCall(PathSegment { foo, [Bar, Baz] }, [x, a, b, c, d])`.
    /// The final `Span` represents the span of the function and arguments
    /// (e.g. `foo::<Bar, Baz>(a, b, c, d)` in `x.foo::<Bar, Baz>(a, b, c, d)`
    ///
    /// To resolve the called method to a `DefId`, call [`type_dependent_def_id`] with
    /// the `hir_id` of the `MethodCall` node itself.
    ///
    /// [`type_dependent_def_id`]: ../ty/struct.TypeckResults.html#method.type_dependent_def_id
    MethodCall(&'hir PathSegment<'hir>, Span, &'hir [Expr<'hir>], Span),
    /// A tuple (e.g., `(a, b, c, d)`).
    Tup(&'hir [Expr<'hir>]),
    /// A binary operation (e.g., `a + b`, `a * b`).
    Binary(BinOp, &'hir Expr<'hir>, &'hir Expr<'hir>),
    /// A unary operation (e.g., `!x`, `*x`).
    Unary(UnOp, &'hir Expr<'hir>),
    /// A literal (e.g., `1`, `"foo"`).
    Lit(Lit),
    /// A cast (e.g., `foo as f64`).
    Cast(&'hir Expr<'hir>, &'hir Ty<'hir>),
    /// A type reference (e.g., `Foo`).
    Type(&'hir Expr<'hir>, &'hir Ty<'hir>),
    /// Wraps the expression in a terminating scope.
    /// This makes it semantically equivalent to `{ let _t = expr; _t }`.
    ///
    /// This construct only exists to tweak the drop order in HIR lowering.
    /// An example of that is the desugaring of `for` loops.
    DropTemps(&'hir Expr<'hir>),
    /// A `let $pat = $expr` expression.
    ///
    /// These are not `Local` and only occur as expressions.
    /// The `let Some(x) = foo()` in `if let Some(x) = foo()` is an example of `Let(..)`.
    Let(&'hir Pat<'hir>, &'hir Expr<'hir>, Span),
    /// An `if` block, with an optional else block.
    ///
    /// I.e., `if <expr> { <expr> } else { <expr> }`.
    If(&'hir Expr<'hir>, &'hir Expr<'hir>, Option<&'hir Expr<'hir>>),
    /// A conditionless loop (can be exited with `break`, `continue`, or `return`).
    ///
    /// I.e., `'label: loop { <block> }`.
    ///
    /// The `Span` is the loop header (`for x in y`/`while let pat = expr`).
    Loop(&'hir Block<'hir>, Option<Label>, LoopSource, Span),
    /// A `match` block, with a source that indicates whether or not it is
    /// the result of a desugaring, and if so, which kind.
    Match(&'hir Expr<'hir>, &'hir [Arm<'hir>], MatchSource),
    /// A closure (e.g., `move |a, b, c| {a + b + c}`).
    ///
    /// The `Span` is the argument block `|...|`.
    ///
    /// This may also be a generator literal or an `async block` as indicated by the
    /// `Option<Movability>`.
    Closure(CaptureBy, &'hir FnDecl<'hir>, BodyId, Span, Option<Movability>),
    /// A block (e.g., `'label: { ... }`).
    Block(&'hir Block<'hir>, Option<Label>),

    /// An assignment (e.g., `a = foo()`).
    Assign(&'hir Expr<'hir>, &'hir Expr<'hir>, Span),
    /// An assignment with an operator.
    ///
    /// E.g., `a += 1`.
    AssignOp(BinOp, &'hir Expr<'hir>, &'hir Expr<'hir>),
    /// Access of a named (e.g., `obj.foo`) or unnamed (e.g., `obj.0`) struct or tuple field.
    Field(&'hir Expr<'hir>, Ident),
    /// An indexing operation (`foo[2]`).
    Index(&'hir Expr<'hir>, &'hir Expr<'hir>),

    /// Path to a definition, possibly containing lifetime or type parameters.
    Path(QPath<'hir>),

    /// A referencing operation (i.e., `&a` or `&mut a`).
    AddrOf(BorrowKind, Mutability, &'hir Expr<'hir>),
    /// A `break`, with an optional label to break.
    Break(Destination, Option<&'hir Expr<'hir>>),
    /// A `continue`, with an optional label.
    Continue(Destination),
    /// A `return`, with an optional value to be returned.
    Ret(Option<&'hir Expr<'hir>>),

    /// Inline assembly (from `asm!`), with its outputs and inputs.
    InlineAsm(&'hir InlineAsm<'hir>),
    /// Inline assembly (from `llvm_asm!`), with its outputs and inputs.
    LlvmInlineAsm(&'hir LlvmInlineAsm<'hir>),

    /// A struct or struct-like variant literal expression.
    ///
    /// E.g., `Foo {x: 1, y: 2}`, or `Foo {x: 1, .. base}`,
    /// where `base` is the `Option<Expr>`.
    Struct(&'hir QPath<'hir>, &'hir [ExprField<'hir>], Option<&'hir Expr<'hir>>),

    /// An array literal constructed from one repeated element.
    ///
    /// E.g., `[1; 5]`. The first expression is the element
    /// to be repeated; the second is the number of times to repeat it.
    Repeat(&'hir Expr<'hir>, AnonConst),

    /// A suspension point for generators (i.e., `yield <expr>`).
    Yield(&'hir Expr<'hir>, YieldSource),

    /// A placeholder for an expression that wasn't syntactically well formed in some way.
    Err,
}

/// Represents an optionally `Self`-qualified value/type path or associated extension.
///
/// To resolve the path to a `DefId`, call [`qpath_res`].
///
/// [`qpath_res`]: ../rustc_middle/ty/struct.TypeckResults.html#method.qpath_res
#[derive(Debug, HashStable_Generic)]
pub enum QPath<'hir> {
    /// Path to a definition, optionally "fully-qualified" with a `Self`
    /// type, if the path points to an associated item in a trait.
    ///
    /// E.g., an unqualified path like `Clone::clone` has `None` for `Self`,
    /// while `<Vec<T> as Clone>::clone` has `Some(Vec<T>)` for `Self`,
    /// even though they both have the same two-segment `Clone::clone` `Path`.
    Resolved(Option<&'hir Ty<'hir>>, &'hir Path<'hir>),

    /// Type-related paths (e.g., `<T>::default` or `<T>::Output`).
    /// Will be resolved by type-checking to an associated item.
    ///
    /// UFCS source paths can desugar into this, with `Vec::new` turning into
    /// `<Vec>::new`, and `T::X::Y::method` into `<<<T>::X>::Y>::method`,
    /// the `X` and `Y` nodes each being a `TyKind::Path(QPath::TypeRelative(..))`.
    TypeRelative(&'hir Ty<'hir>, &'hir PathSegment<'hir>),

    /// Reference to a `#[lang = "foo"]` item.
    LangItem(LangItem, Span),
}

impl<'hir> QPath<'hir> {
    /// Returns the span of this `QPath`.
    pub fn span(&self) -> Span {
        match *self {
            QPath::Resolved(_, path) => path.span,
            QPath::TypeRelative(qself, ps) => qself.span.to(ps.ident.span),
            QPath::LangItem(_, span) => span,
        }
    }

    /// Returns the span of the qself of this `QPath`. For example, `()` in
    /// `<() as Trait>::method`.
    pub fn qself_span(&self) -> Span {
        match *self {
            QPath::Resolved(_, path) => path.span,
            QPath::TypeRelative(qself, _) => qself.span,
            QPath::LangItem(_, span) => span,
        }
    }

    /// Returns the span of the last segment of this `QPath`. For example, `method` in
    /// `<() as Trait>::method`.
    pub fn last_segment_span(&self) -> Span {
        match *self {
            QPath::Resolved(_, path) => path.segments.last().unwrap().ident.span,
            QPath::TypeRelative(_, segment) => segment.ident.span,
            QPath::LangItem(_, span) => span,
        }
    }
}

/// Hints at the original code for a let statement.
#[derive(Copy, Clone, Encodable, Debug, HashStable_Generic)]
pub enum LocalSource {
    /// A `match _ { .. }`.
    Normal,
    /// A desugared `for _ in _ { .. }` loop.
    ForLoopDesugar,
    /// When lowering async functions, we create locals within the `async move` so that
    /// all parameters are dropped after the future is polled.
    ///
    /// ```ignore (pseudo-Rust)
    /// async fn foo(<pattern> @ x: Type) {
    ///     async move {
    ///         let <pattern> = x;
    ///     }
    /// }
    /// ```
    AsyncFn,
    /// A desugared `<expr>.await`.
    AwaitDesugar,
    /// A desugared `expr = expr`, where the LHS is a tuple, struct or array.
    /// The span is that of the `=` sign.
    AssignDesugar(Span),
}

/// Hints at the original code for a `match _ { .. }`.
#[derive(Copy, Clone, PartialEq, Eq, Encodable, Hash, Debug)]
#[derive(HashStable_Generic)]
pub enum MatchSource {
    /// A `match _ { .. }`.
    Normal,
    /// A desugared `for _ in _ { .. }` loop.
    ForLoopDesugar,
    /// A desugared `?` operator.
    TryDesugar,
    /// A desugared `<expr>.await`.
    AwaitDesugar,
}

impl MatchSource {
    #[inline]
    pub const fn name(self) -> &'static str {
        use MatchSource::*;
        match self {
            Normal => "match",
            ForLoopDesugar => "for",
            TryDesugar => "?",
            AwaitDesugar => ".await",
        }
    }
}

/// The loop type that yielded an `ExprKind::Loop`.
#[derive(Copy, Clone, PartialEq, Encodable, Debug, HashStable_Generic)]
pub enum LoopSource {
    /// A `loop { .. }` loop.
    Loop,
    /// A `while _ { .. }` loop.
    While,
    /// A `for _ in _ { .. }` loop.
    ForLoop,
}

impl LoopSource {
    pub fn name(self) -> &'static str {
        match self {
            LoopSource::Loop => "loop",
            LoopSource::While => "while",
            LoopSource::ForLoop => "for",
        }
    }
}

#[derive(Copy, Clone, Encodable, Debug, HashStable_Generic)]
pub enum LoopIdError {
    OutsideLoopScope,
    UnlabeledCfInWhileCondition,
    UnresolvedLabel,
}

impl fmt::Display for LoopIdError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            LoopIdError::OutsideLoopScope => "not inside loop scope",
            LoopIdError::UnlabeledCfInWhileCondition => {
                "unlabeled control flow (break or continue) in while condition"
            }
            LoopIdError::UnresolvedLabel => "label not found",
        })
    }
}

#[derive(Copy, Clone, Encodable, Debug, HashStable_Generic)]
pub struct Destination {
    // This is `Some(_)` iff there is an explicit user-specified `label
    pub label: Option<Label>,

    // These errors are caught and then reported during the diagnostics pass in
    // librustc_passes/loops.rs
    pub target_id: Result<HirId, LoopIdError>,
}

/// The yield kind that caused an `ExprKind::Yield`.
#[derive(Copy, Clone, PartialEq, Eq, Debug, Encodable, Decodable, HashStable_Generic)]
pub enum YieldSource {
    /// An `<expr>.await`.
    Await { expr: Option<HirId> },
    /// A plain `yield`.
    Yield,
}

impl YieldSource {
    pub fn is_await(&self) -> bool {
        match self {
            YieldSource::Await { .. } => true,
            YieldSource::Yield => false,
        }
    }
}

impl fmt::Display for YieldSource {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            YieldSource::Await { .. } => "`await`",
            YieldSource::Yield => "`yield`",
        })
    }
}

impl From<GeneratorKind> for YieldSource {
    fn from(kind: GeneratorKind) -> Self {
        match kind {
            // Guess based on the kind of the current generator.
            GeneratorKind::Gen => Self::Yield,
            GeneratorKind::Async(_) => Self::Await { expr: None },
        }
    }
}

// N.B., if you change this, you'll probably want to change the corresponding
// type structure in middle/ty.rs as well.
#[derive(Debug, HashStable_Generic)]
pub struct MutTy<'hir> {
    pub ty: &'hir Ty<'hir>,
    pub mutbl: Mutability,
}

/// Represents a function's signature in a trait declaration,
/// trait implementation, or a free function.
#[derive(Debug, HashStable_Generic)]
pub struct FnSig<'hir> {
    pub header: FnHeader,
    pub decl: &'hir FnDecl<'hir>,
    pub span: Span,
}

// The bodies for items are stored "out of line", in a separate
// hashmap in the `Crate`. Here we just record the hir-id of the item
// so it can fetched later.
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Encodable, Debug)]
pub struct TraitItemId {
    pub def_id: LocalDefId,
}

impl TraitItemId {
    #[inline]
    pub fn hir_id(&self) -> HirId {
        // Items are always HIR owners.
        HirId::make_owner(self.def_id)
    }
}

/// Represents an item declaration within a trait declaration,
/// possibly including a default implementation. A trait item is
/// either required (meaning it doesn't have an implementation, just a
/// signature) or provided (meaning it has a default implementation).
#[derive(Debug)]
pub struct TraitItem<'hir> {
    pub ident: Ident,
    pub def_id: LocalDefId,
    pub generics: Generics<'hir>,
    pub kind: TraitItemKind<'hir>,
    pub span: Span,
}

impl TraitItem<'_> {
    #[inline]
    pub fn hir_id(&self) -> HirId {
        // Items are always HIR owners.
        HirId::make_owner(self.def_id)
    }

    pub fn trait_item_id(&self) -> TraitItemId {
        TraitItemId { def_id: self.def_id }
    }
}

/// Represents a trait method's body (or just argument names).
#[derive(Encodable, Debug, HashStable_Generic)]
pub enum TraitFn<'hir> {
    /// No default body in the trait, just a signature.
    Required(&'hir [Ident]),

    /// Both signature and body are provided in the trait.
    Provided(BodyId),
}

/// Represents a trait method or associated constant or type
#[derive(Debug, HashStable_Generic)]
pub enum TraitItemKind<'hir> {
    /// An associated constant with an optional value (otherwise `impl`s must contain a value).
    Const(&'hir Ty<'hir>, Option<BodyId>),
    /// An associated function with an optional body.
    Fn(FnSig<'hir>, TraitFn<'hir>),
    /// An associated type with (possibly empty) bounds and optional concrete
    /// type.
    Type(GenericBounds<'hir>, Option<&'hir Ty<'hir>>),
}

// The bodies for items are stored "out of line", in a separate
// hashmap in the `Crate`. Here we just record the hir-id of the item
// so it can fetched later.
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Encodable, Debug)]
pub struct ImplItemId {
    pub def_id: LocalDefId,
}

impl ImplItemId {
    #[inline]
    pub fn hir_id(&self) -> HirId {
        // Items are always HIR owners.
        HirId::make_owner(self.def_id)
    }
}

/// Represents anything within an `impl` block.
#[derive(Debug)]
pub struct ImplItem<'hir> {
    pub ident: Ident,
    pub def_id: LocalDefId,
    pub vis: Visibility<'hir>,
    pub defaultness: Defaultness,
    pub generics: Generics<'hir>,
    pub kind: ImplItemKind<'hir>,
    pub span: Span,
}

impl ImplItem<'_> {
    #[inline]
    pub fn hir_id(&self) -> HirId {
        // Items are always HIR owners.
        HirId::make_owner(self.def_id)
    }

    pub fn impl_item_id(&self) -> ImplItemId {
        ImplItemId { def_id: self.def_id }
    }
}

/// Represents various kinds of content within an `impl`.
#[derive(Debug, HashStable_Generic)]
pub enum ImplItemKind<'hir> {
    /// An associated constant of the given type, set to the constant result
    /// of the expression.
    Const(&'hir Ty<'hir>, BodyId),
    /// An associated function implementation with the given signature and body.
    Fn(FnSig<'hir>, BodyId),
    /// An associated type.
    TyAlias(&'hir Ty<'hir>),
}

// The name of the associated type for `Fn` return types.
pub const FN_OUTPUT_NAME: Symbol = sym::Output;

/// Bind a type to an associated type (i.e., `A = Foo`).
///
/// Bindings like `A: Debug` are represented as a special type `A =
/// $::Debug` that is understood by the astconv code.
///
/// FIXME(alexreg): why have a separate type for the binding case,
/// wouldn't it be better to make the `ty` field an enum like the
/// following?
///
/// ```
/// enum TypeBindingKind {
///    Equals(...),
///    Binding(...),
/// }
/// ```
#[derive(Debug, HashStable_Generic)]
pub struct TypeBinding<'hir> {
    pub hir_id: HirId,
    #[stable_hasher(project(name))]
    pub ident: Ident,
    pub gen_args: &'hir GenericArgs<'hir>,
    pub kind: TypeBindingKind<'hir>,
    pub span: Span,
}

// Represents the two kinds of type bindings.
#[derive(Debug, HashStable_Generic)]
pub enum TypeBindingKind<'hir> {
    /// E.g., `Foo<Bar: Send>`.
    Constraint { bounds: &'hir [GenericBound<'hir>] },
    /// E.g., `Foo<Bar = ()>`.
    Equality { ty: &'hir Ty<'hir> },
}

impl TypeBinding<'_> {
    pub fn ty(&self) -> &Ty<'_> {
        match self.kind {
            TypeBindingKind::Equality { ref ty } => ty,
            _ => panic!("expected equality type binding for parenthesized generic args"),
        }
    }
}

#[derive(Debug)]
pub struct Ty<'hir> {
    pub hir_id: HirId,
    pub kind: TyKind<'hir>,
    pub span: Span,
}

/// Not represented directly in the AST; referred to by name through a `ty_path`.
#[derive(Copy, Clone, PartialEq, Eq, Encodable, Decodable, Hash, Debug)]
#[derive(HashStable_Generic)]
pub enum PrimTy {
    Int(IntTy),
    Uint(UintTy),
    Float(FloatTy),
    Str,
    Bool,
    Char,
}

impl PrimTy {
    /// All of the primitive types
    pub const ALL: [Self; 17] = [
        // any changes here should also be reflected in `PrimTy::from_name`
        Self::Int(IntTy::I8),
        Self::Int(IntTy::I16),
        Self::Int(IntTy::I32),
        Self::Int(IntTy::I64),
        Self::Int(IntTy::I128),
        Self::Int(IntTy::Isize),
        Self::Uint(UintTy::U8),
        Self::Uint(UintTy::U16),
        Self::Uint(UintTy::U32),
        Self::Uint(UintTy::U64),
        Self::Uint(UintTy::U128),
        Self::Uint(UintTy::Usize),
        Self::Float(FloatTy::F32),
        Self::Float(FloatTy::F64),
        Self::Bool,
        Self::Char,
        Self::Str,
    ];

    /// Like [`PrimTy::name`], but returns a &str instead of a symbol.
    ///
    /// Used by clippy.
    pub fn name_str(self) -> &'static str {
        match self {
            PrimTy::Int(i) => i.name_str(),
            PrimTy::Uint(u) => u.name_str(),
            PrimTy::Float(f) => f.name_str(),
            PrimTy::Str => "str",
            PrimTy::Bool => "bool",
            PrimTy::Char => "char",
        }
    }

    pub fn name(self) -> Symbol {
        match self {
            PrimTy::Int(i) => i.name(),
            PrimTy::Uint(u) => u.name(),
            PrimTy::Float(f) => f.name(),
            PrimTy::Str => sym::str,
            PrimTy::Bool => sym::bool,
            PrimTy::Char => sym::char,
        }
    }

    /// Returns the matching `PrimTy` for a `Symbol` such as "str" or "i32".
    /// Returns `None` if no matching type is found.
    pub fn from_name(name: Symbol) -> Option<Self> {
        let ty = match name {
            // any changes here should also be reflected in `PrimTy::ALL`
            sym::i8 => Self::Int(IntTy::I8),
            sym::i16 => Self::Int(IntTy::I16),
            sym::i32 => Self::Int(IntTy::I32),
            sym::i64 => Self::Int(IntTy::I64),
            sym::i128 => Self::Int(IntTy::I128),
            sym::isize => Self::Int(IntTy::Isize),
            sym::u8 => Self::Uint(UintTy::U8),
            sym::u16 => Self::Uint(UintTy::U16),
            sym::u32 => Self::Uint(UintTy::U32),
            sym::u64 => Self::Uint(UintTy::U64),
            sym::u128 => Self::Uint(UintTy::U128),
            sym::usize => Self::Uint(UintTy::Usize),
            sym::f32 => Self::Float(FloatTy::F32),
            sym::f64 => Self::Float(FloatTy::F64),
            sym::bool => Self::Bool,
            sym::char => Self::Char,
            sym::str => Self::Str,
            _ => return None,
        };
        Some(ty)
    }
}

#[derive(Debug, HashStable_Generic)]
pub struct BareFnTy<'hir> {
    pub unsafety: Unsafety,
    pub abi: Abi,
    pub generic_params: &'hir [GenericParam<'hir>],
    pub decl: &'hir FnDecl<'hir>,
    pub param_names: &'hir [Ident],
}

#[derive(Debug, HashStable_Generic)]
pub struct OpaqueTy<'hir> {
    pub generics: Generics<'hir>,
    pub bounds: GenericBounds<'hir>,
    pub impl_trait_fn: Option<DefId>,
    pub origin: OpaqueTyOrigin,
}

/// From whence the opaque type came.
#[derive(Copy, Clone, PartialEq, Eq, Encodable, Decodable, Debug, HashStable_Generic)]
pub enum OpaqueTyOrigin {
    /// `-> impl Trait`
    FnReturn,
    /// `async fn`
    AsyncFn,
    /// type aliases: `type Foo = impl Trait;`
    TyAlias,
}

/// The various kinds of types recognized by the compiler.
#[derive(Debug, HashStable_Generic)]
pub enum TyKind<'hir> {
    /// A variable length slice (i.e., `[T]`).
    Slice(&'hir Ty<'hir>),
    /// A fixed length array (i.e., `[T; n]`).
    Array(&'hir Ty<'hir>, AnonConst),
    /// A raw pointer (i.e., `*const T` or `*mut T`).
    Ptr(MutTy<'hir>),
    /// A reference (i.e., `&'a T` or `&'a mut T`).
    Rptr(Lifetime, MutTy<'hir>),
    /// A bare function (e.g., `fn(usize) -> bool`).
    BareFn(&'hir BareFnTy<'hir>),
    /// The never type (`!`).
    Never,
    /// A tuple (`(A, B, C, D, ...)`).
    Tup(&'hir [Ty<'hir>]),
    /// A path to a type definition (`module::module::...::Type`), or an
    /// associated type (e.g., `<Vec<T> as Trait>::Type` or `<T>::Target`).
    ///
    /// Type parameters may be stored in each `PathSegment`.
    Path(QPath<'hir>),
    /// An opaque type definition itself. This is currently only used for the
    /// `opaque type Foo: Trait` item that `impl Trait` in desugars to.
    ///
    /// The generic argument list contains the lifetimes (and in the future
    /// possibly parameters) that are actually bound on the `impl Trait`.
    OpaqueDef(ItemId, &'hir [GenericArg<'hir>]),
    /// A trait object type `Bound1 + Bound2 + Bound3`
    /// where `Bound` is a trait or a lifetime.
    TraitObject(&'hir [PolyTraitRef<'hir>], Lifetime, TraitObjectSyntax),
    /// Unused for now.
    Typeof(AnonConst),
    /// `TyKind::Infer` means the type should be inferred instead of it having been
    /// specified. This can appear anywhere in a type.
    Infer,
    /// Placeholder for a type that has failed to be defined.
    Err,
}

#[derive(Debug, HashStable_Generic)]
pub enum InlineAsmOperand<'hir> {
    In {
        reg: InlineAsmRegOrRegClass,
        expr: Expr<'hir>,
    },
    Out {
        reg: InlineAsmRegOrRegClass,
        late: bool,
        expr: Option<Expr<'hir>>,
    },
    InOut {
        reg: InlineAsmRegOrRegClass,
        late: bool,
        expr: Expr<'hir>,
    },
    SplitInOut {
        reg: InlineAsmRegOrRegClass,
        late: bool,
        in_expr: Expr<'hir>,
        out_expr: Option<Expr<'hir>>,
    },
    Const {
        anon_const: AnonConst,
    },
    Sym {
        expr: Expr<'hir>,
    },
}

impl<'hir> InlineAsmOperand<'hir> {
    pub fn reg(&self) -> Option<InlineAsmRegOrRegClass> {
        match *self {
            Self::In { reg, .. }
            | Self::Out { reg, .. }
            | Self::InOut { reg, .. }
            | Self::SplitInOut { reg, .. } => Some(reg),
            Self::Const { .. } | Self::Sym { .. } => None,
        }
    }
}

#[derive(Debug, HashStable_Generic)]
pub struct InlineAsm<'hir> {
    pub template: &'hir [InlineAsmTemplatePiece],
    pub template_strs: &'hir [(Symbol, Option<Symbol>, Span)],
    pub operands: &'hir [(InlineAsmOperand<'hir>, Span)],
    pub options: InlineAsmOptions,
    pub line_spans: &'hir [Span],
}

#[derive(Copy, Clone, Encodable, Decodable, Debug, Hash, HashStable_Generic, PartialEq)]
pub struct LlvmInlineAsmOutput {
    pub constraint: Symbol,
    pub is_rw: bool,
    pub is_indirect: bool,
    pub span: Span,
}

// NOTE(eddyb) This is used within MIR as well, so unlike the rest of the HIR,
// it needs to be `Clone` and `Decodable` and use plain `Vec<T>` instead of
// arena-allocated slice.
#[derive(Clone, Encodable, Decodable, Debug, Hash, HashStable_Generic, PartialEq)]
pub struct LlvmInlineAsmInner {
    pub asm: Symbol,
    pub asm_str_style: StrStyle,
    pub outputs: Vec<LlvmInlineAsmOutput>,
    pub inputs: Vec<Symbol>,
    pub clobbers: Vec<Symbol>,
    pub volatile: bool,
    pub alignstack: bool,
    pub dialect: LlvmAsmDialect,
}

#[derive(Debug, HashStable_Generic)]
pub struct LlvmInlineAsm<'hir> {
    pub inner: LlvmInlineAsmInner,
    pub outputs_exprs: &'hir [Expr<'hir>],
    pub inputs_exprs: &'hir [Expr<'hir>],
}

/// Represents a parameter in a function header.
#[derive(Debug, HashStable_Generic)]
pub struct Param<'hir> {
    pub hir_id: HirId,
    pub pat: &'hir Pat<'hir>,
    pub ty_span: Span,
    pub span: Span,
}

/// Represents the header (not the body) of a function declaration.
#[derive(Debug, HashStable_Generic)]
pub struct FnDecl<'hir> {
    /// The types of the function's parameters.
    ///
    /// Additional argument data is stored in the function's [body](Body::params).
    pub inputs: &'hir [Ty<'hir>],
    pub output: FnRetTy<'hir>,
    pub c_variadic: bool,
    /// Does the function have an implicit self?
    pub implicit_self: ImplicitSelfKind,
}

/// Represents what type of implicit self a function has, if any.
#[derive(Copy, Clone, Encodable, Decodable, Debug, HashStable_Generic)]
pub enum ImplicitSelfKind {
    /// Represents a `fn x(self);`.
    Imm,
    /// Represents a `fn x(mut self);`.
    Mut,
    /// Represents a `fn x(&self);`.
    ImmRef,
    /// Represents a `fn x(&mut self);`.
    MutRef,
    /// Represents when a function does not have a self argument or
    /// when a function has a `self: X` argument.
    None,
}

impl ImplicitSelfKind {
    /// Does this represent an implicit self?
    pub fn has_implicit_self(&self) -> bool {
        !matches!(*self, ImplicitSelfKind::None)
    }
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Encodable, Decodable, Debug)]
#[derive(HashStable_Generic)]
pub enum IsAsync {
    Async,
    NotAsync,
}

#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug, Encodable, Decodable, HashStable_Generic)]
pub enum Defaultness {
    Default { has_value: bool },
    Final,
}

impl Defaultness {
    pub fn has_value(&self) -> bool {
        match *self {
            Defaultness::Default { has_value } => has_value,
            Defaultness::Final => true,
        }
    }

    pub fn is_final(&self) -> bool {
        *self == Defaultness::Final
    }

    pub fn is_default(&self) -> bool {
        matches!(*self, Defaultness::Default { .. })
    }
}

#[derive(Debug, HashStable_Generic)]
pub enum FnRetTy<'hir> {
    /// Return type is not specified.
    ///
    /// Functions default to `()` and
    /// closures default to inference. Span points to where return
    /// type would be inserted.
    DefaultReturn(Span),
    /// Everything else.
    Return(&'hir Ty<'hir>),
}

impl FnRetTy<'_> {
    #[inline]
    pub fn span(&self) -> Span {
        match *self {
            Self::DefaultReturn(span) => span,
            Self::Return(ref ty) => ty.span,
        }
    }
}

#[derive(Encodable, Debug)]
pub struct Mod<'hir> {
    /// A span from the first token past `{` to the last token until `}`.
    /// For `mod foo;`, the inner span ranges from the first token
    /// to the last token in the external file.
    pub inner: Span,
    pub item_ids: &'hir [ItemId],
}

#[derive(Debug, HashStable_Generic)]
pub struct EnumDef<'hir> {
    pub variants: &'hir [Variant<'hir>],
}

#[derive(Debug, HashStable_Generic)]
pub struct Variant<'hir> {
    /// Name of the variant.
    #[stable_hasher(project(name))]
    pub ident: Ident,
    /// Id of the variant (not the constructor, see `VariantData::ctor_hir_id()`).
    pub id: HirId,
    /// Fields and constructor id of the variant.
    pub data: VariantData<'hir>,
    /// Explicit discriminant (e.g., `Foo = 1`).
    pub disr_expr: Option<AnonConst>,
    /// Span
    pub span: Span,
}

#[derive(Copy, Clone, PartialEq, Encodable, Debug, HashStable_Generic)]
pub enum UseKind {
    /// One import, e.g., `use foo::bar` or `use foo::bar as baz`.
    /// Also produced for each element of a list `use`, e.g.
    /// `use foo::{a, b}` lowers to `use foo::a; use foo::b;`.
    Single,

    /// Glob import, e.g., `use foo::*`.
    Glob,

    /// Degenerate list import, e.g., `use foo::{a, b}` produces
    /// an additional `use foo::{}` for performing checks such as
    /// unstable feature gating. May be removed in the future.
    ListStem,
}

/// References to traits in impls.
///
/// `resolve` maps each `TraitRef`'s `ref_id` to its defining trait; that's all
/// that the `ref_id` is for. Note that `ref_id`'s value is not the `HirId` of the
/// trait being referred to but just a unique `HirId` that serves as a key
/// within the resolution map.
#[derive(Clone, Debug, HashStable_Generic)]
pub struct TraitRef<'hir> {
    pub path: &'hir Path<'hir>,
    // Don't hash the `ref_id`. It is tracked via the thing it is used to access.
    #[stable_hasher(ignore)]
    pub hir_ref_id: HirId,
}

impl TraitRef<'_> {
    /// Gets the `DefId` of the referenced trait. It _must_ actually be a trait or trait alias.
    pub fn trait_def_id(&self) -> Option<DefId> {
        match self.path.res {
            Res::Def(DefKind::Trait | DefKind::TraitAlias, did) => Some(did),
            Res::Err => None,
            _ => unreachable!(),
        }
    }
}

#[derive(Clone, Debug, HashStable_Generic)]
pub struct PolyTraitRef<'hir> {
    /// The `'a` in `for<'a> Foo<&'a T>`.
    pub bound_generic_params: &'hir [GenericParam<'hir>],

    /// The `Foo<&'a T>` in `for<'a> Foo<&'a T>`.
    pub trait_ref: TraitRef<'hir>,

    pub span: Span,
}

pub type Visibility<'hir> = Spanned<VisibilityKind<'hir>>;

#[derive(Copy, Clone, Debug)]
pub enum VisibilityKind<'hir> {
    Public,
    Crate(CrateSugar),
    Restricted { path: &'hir Path<'hir>, hir_id: HirId },
    Inherited,
}

impl VisibilityKind<'_> {
    pub fn is_pub(&self) -> bool {
        matches!(*self, VisibilityKind::Public)
    }

    pub fn is_pub_restricted(&self) -> bool {
        match *self {
            VisibilityKind::Public | VisibilityKind::Inherited => false,
            VisibilityKind::Crate(..) | VisibilityKind::Restricted { .. } => true,
        }
    }
}

#[derive(Debug, HashStable_Generic)]
pub struct FieldDef<'hir> {
    pub span: Span,
    #[stable_hasher(project(name))]
    pub ident: Ident,
    pub vis: Visibility<'hir>,
    pub hir_id: HirId,
    pub ty: &'hir Ty<'hir>,
}

impl FieldDef<'_> {
    // Still necessary in couple of places
    pub fn is_positional(&self) -> bool {
        let first = self.ident.as_str().as_bytes()[0];
        (b'0'..=b'9').contains(&first)
    }
}

/// Fields and constructor IDs of enum variants and structs.
#[derive(Debug, HashStable_Generic)]
pub enum VariantData<'hir> {
    /// A struct variant.
    ///
    /// E.g., `Bar { .. }` as in `enum Foo { Bar { .. } }`.
    Struct(&'hir [FieldDef<'hir>], /* recovered */ bool),
    /// A tuple variant.
    ///
    /// E.g., `Bar(..)` as in `enum Foo { Bar(..) }`.
    Tuple(&'hir [FieldDef<'hir>], HirId),
    /// A unit variant.
    ///
    /// E.g., `Bar = ..` as in `enum Foo { Bar = .. }`.
    Unit(HirId),
}

impl VariantData<'hir> {
    /// Return the fields of this variant.
    pub fn fields(&self) -> &'hir [FieldDef<'hir>] {
        match *self {
            VariantData::Struct(ref fields, ..) | VariantData::Tuple(ref fields, ..) => fields,
            _ => &[],
        }
    }

    /// Return the `HirId` of this variant's constructor, if it has one.
    pub fn ctor_hir_id(&self) -> Option<HirId> {
        match *self {
            VariantData::Struct(_, _) => None,
            VariantData::Tuple(_, hir_id) | VariantData::Unit(hir_id) => Some(hir_id),
        }
    }
}

// The bodies for items are stored "out of line", in a separate
// hashmap in the `Crate`. Here we just record the hir-id of the item
// so it can fetched later.
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Encodable, Debug, Hash)]
pub struct ItemId {
    pub def_id: LocalDefId,
}

impl ItemId {
    #[inline]
    pub fn hir_id(&self) -> HirId {
        // Items are always HIR owners.
        HirId::make_owner(self.def_id)
    }
}

/// An item
///
/// The name might be a dummy name in case of anonymous items
#[derive(Debug)]
pub struct Item<'hir> {
    pub ident: Ident,
    pub def_id: LocalDefId,
    pub kind: ItemKind<'hir>,
    pub vis: Visibility<'hir>,
    pub span: Span,
}

impl Item<'_> {
    #[inline]
    pub fn hir_id(&self) -> HirId {
        // Items are always HIR owners.
        HirId::make_owner(self.def_id)
    }

    pub fn item_id(&self) -> ItemId {
        ItemId { def_id: self.def_id }
    }
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
#[derive(Encodable, Decodable, HashStable_Generic)]
pub enum Unsafety {
    Unsafe,
    Normal,
}

impl Unsafety {
    pub fn prefix_str(&self) -> &'static str {
        match self {
            Self::Unsafe => "unsafe ",
            Self::Normal => "",
        }
    }
}

impl fmt::Display for Unsafety {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match *self {
            Self::Unsafe => "unsafe",
            Self::Normal => "normal",
        })
    }
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
#[derive(Encodable, Decodable, HashStable_Generic)]
pub enum Constness {
    Const,
    NotConst,
}

impl fmt::Display for Constness {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match *self {
            Self::Const => "const",
            Self::NotConst => "non-const",
        })
    }
}

#[derive(Copy, Clone, Encodable, Debug, HashStable_Generic)]
pub struct FnHeader {
    pub unsafety: Unsafety,
    pub constness: Constness,
    pub asyncness: IsAsync,
    pub abi: Abi,
}

impl FnHeader {
    pub fn is_const(&self) -> bool {
        matches!(&self.constness, Constness::Const)
    }
}

#[derive(Debug, HashStable_Generic)]
pub enum ItemKind<'hir> {
    /// An `extern crate` item, with optional *original* crate name if the crate was renamed.
    ///
    /// E.g., `extern crate foo` or `extern crate foo_bar as foo`.
    ExternCrate(Option<Symbol>),

    /// `use foo::bar::*;` or `use foo::bar::baz as quux;`
    ///
    /// or just
    ///
    /// `use foo::bar::baz;` (with `as baz` implicitly on the right).
    Use(&'hir Path<'hir>, UseKind),

    /// A `static` item.
    Static(&'hir Ty<'hir>, Mutability, BodyId),
    /// A `const` item.
    Const(&'hir Ty<'hir>, BodyId),
    /// A function declaration.
    Fn(FnSig<'hir>, Generics<'hir>, BodyId),
    /// A MBE macro definition (`macro_rules!` or `macro`).
    Macro(ast::MacroDef),
    /// A module.
    Mod(Mod<'hir>),
    /// An external module, e.g. `extern { .. }`.
    ForeignMod { abi: Abi, items: &'hir [ForeignItemRef] },
    /// Module-level inline assembly (from `global_asm!`).
    GlobalAsm(&'hir InlineAsm<'hir>),
    /// A type alias, e.g., `type Foo = Bar<u8>`.
    TyAlias(&'hir Ty<'hir>, Generics<'hir>),
    /// An opaque `impl Trait` type alias, e.g., `type Foo = impl Bar;`.
    OpaqueTy(OpaqueTy<'hir>),
    /// An enum definition, e.g., `enum Foo<A, B> {C<A>, D<B>}`.
    Enum(EnumDef<'hir>, Generics<'hir>),
    /// A struct definition, e.g., `struct Foo<A> {x: A}`.
    Struct(VariantData<'hir>, Generics<'hir>),
    /// A union definition, e.g., `union Foo<A, B> {x: A, y: B}`.
    Union(VariantData<'hir>, Generics<'hir>),
    /// A trait definition.
    Trait(IsAuto, Unsafety, Generics<'hir>, GenericBounds<'hir>, &'hir [TraitItemRef]),
    /// A trait alias.
    TraitAlias(Generics<'hir>, GenericBounds<'hir>),

    /// An implementation, e.g., `impl<A> Trait for Foo { .. }`.
    Impl(Impl<'hir>),
}

#[derive(Debug, HashStable_Generic)]
pub struct Impl<'hir> {
    pub unsafety: Unsafety,
    pub polarity: ImplPolarity,
    pub defaultness: Defaultness,
    // We do not put a `Span` in `Defaultness` because it breaks foreign crate metadata
    // decoding as `Span`s cannot be decoded when a `Session` is not available.
    pub defaultness_span: Option<Span>,
    pub constness: Constness,
    pub generics: Generics<'hir>,

    /// The trait being implemented, if any.
    pub of_trait: Option<TraitRef<'hir>>,

    pub self_ty: &'hir Ty<'hir>,
    pub items: &'hir [ImplItemRef],
}

impl ItemKind<'_> {
    pub fn generics(&self) -> Option<&Generics<'_>> {
        Some(match *self {
            ItemKind::Fn(_, ref generics, _)
            | ItemKind::TyAlias(_, ref generics)
            | ItemKind::OpaqueTy(OpaqueTy { ref generics, impl_trait_fn: None, .. })
            | ItemKind::Enum(_, ref generics)
            | ItemKind::Struct(_, ref generics)
            | ItemKind::Union(_, ref generics)
            | ItemKind::Trait(_, _, ref generics, _, _)
            | ItemKind::Impl(Impl { ref generics, .. }) => generics,
            _ => return None,
        })
    }

    pub fn descr(&self) -> &'static str {
        match self {
            ItemKind::ExternCrate(..) => "extern crate",
            ItemKind::Use(..) => "`use` import",
            ItemKind::Static(..) => "static item",
            ItemKind::Const(..) => "constant item",
            ItemKind::Fn(..) => "function",
            ItemKind::Macro(..) => "macro",
            ItemKind::Mod(..) => "module",
            ItemKind::ForeignMod { .. } => "extern block",
            ItemKind::GlobalAsm(..) => "global asm item",
            ItemKind::TyAlias(..) => "type alias",
            ItemKind::OpaqueTy(..) => "opaque type",
            ItemKind::Enum(..) => "enum",
            ItemKind::Struct(..) => "struct",
            ItemKind::Union(..) => "union",
            ItemKind::Trait(..) => "trait",
            ItemKind::TraitAlias(..) => "trait alias",
            ItemKind::Impl(..) => "implementation",
        }
    }
}

/// A reference from an trait to one of its associated items. This
/// contains the item's id, naturally, but also the item's name and
/// some other high-level details (like whether it is an associated
/// type or method, and whether it is public). This allows other
/// passes to find the impl they want without loading the ID (which
/// means fewer edges in the incremental compilation graph).
#[derive(Encodable, Debug, HashStable_Generic)]
pub struct TraitItemRef {
    pub id: TraitItemId,
    #[stable_hasher(project(name))]
    pub ident: Ident,
    pub kind: AssocItemKind,
    pub span: Span,
    pub defaultness: Defaultness,
}

/// A reference from an impl to one of its associated items. This
/// contains the item's ID, naturally, but also the item's name and
/// some other high-level details (like whether it is an associated
/// type or method, and whether it is public). This allows other
/// passes to find the impl they want without loading the ID (which
/// means fewer edges in the incremental compilation graph).
#[derive(Debug, HashStable_Generic)]
pub struct ImplItemRef {
    pub id: ImplItemId,
    #[stable_hasher(project(name))]
    pub ident: Ident,
    pub kind: AssocItemKind,
    pub span: Span,
    pub defaultness: Defaultness,
}

#[derive(Copy, Clone, PartialEq, Encodable, Debug, HashStable_Generic)]
pub enum AssocItemKind {
    Const,
    Fn { has_self: bool },
    Type,
}

// The bodies for items are stored "out of line", in a separate
// hashmap in the `Crate`. Here we just record the hir-id of the item
// so it can fetched later.
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Encodable, Debug)]
pub struct ForeignItemId {
    pub def_id: LocalDefId,
}

impl ForeignItemId {
    #[inline]
    pub fn hir_id(&self) -> HirId {
        // Items are always HIR owners.
        HirId::make_owner(self.def_id)
    }
}

/// A reference from a foreign block to one of its items. This
/// contains the item's ID, naturally, but also the item's name and
/// some other high-level details (like whether it is an associated
/// type or method, and whether it is public). This allows other
/// passes to find the impl they want without loading the ID (which
/// means fewer edges in the incremental compilation graph).
#[derive(Debug, HashStable_Generic)]
pub struct ForeignItemRef {
    pub id: ForeignItemId,
    #[stable_hasher(project(name))]
    pub ident: Ident,
    pub span: Span,
}

#[derive(Debug)]
pub struct ForeignItem<'hir> {
    pub ident: Ident,
    pub kind: ForeignItemKind<'hir>,
    pub def_id: LocalDefId,
    pub span: Span,
    pub vis: Visibility<'hir>,
}

impl ForeignItem<'_> {
    #[inline]
    pub fn hir_id(&self) -> HirId {
        // Items are always HIR owners.
        HirId::make_owner(self.def_id)
    }

    pub fn foreign_item_id(&self) -> ForeignItemId {
        ForeignItemId { def_id: self.def_id }
    }
}

/// An item within an `extern` block.
#[derive(Debug, HashStable_Generic)]
pub enum ForeignItemKind<'hir> {
    /// A foreign function.
    Fn(&'hir FnDecl<'hir>, &'hir [Ident], Generics<'hir>),
    /// A foreign static item (`static ext: u8`).
    Static(&'hir Ty<'hir>, Mutability),
    /// A foreign type.
    Type,
}

/// A variable captured by a closure.
#[derive(Debug, Copy, Clone, Encodable, HashStable_Generic)]
pub struct Upvar {
    // First span where it is accessed (there can be multiple).
    pub span: Span,
}

// The TraitCandidate's import_ids is empty if the trait is defined in the same module, and
// has length > 0 if the trait is found through an chain of imports, starting with the
// import/use statement in the scope where the trait is used.
#[derive(Encodable, Decodable, Clone, Debug)]
pub struct TraitCandidate {
    pub def_id: DefId,
    pub import_ids: SmallVec<[LocalDefId; 1]>,
}

#[derive(Copy, Clone, Debug, HashStable_Generic)]
pub enum OwnerNode<'hir> {
    Item(&'hir Item<'hir>),
    ForeignItem(&'hir ForeignItem<'hir>),
    TraitItem(&'hir TraitItem<'hir>),
    ImplItem(&'hir ImplItem<'hir>),
    Crate(&'hir Mod<'hir>),
}

impl<'hir> OwnerNode<'hir> {
    pub fn ident(&self) -> Option<Ident> {
        match self {
            OwnerNode::Item(Item { ident, .. })
            | OwnerNode::ForeignItem(ForeignItem { ident, .. })
            | OwnerNode::ImplItem(ImplItem { ident, .. })
            | OwnerNode::TraitItem(TraitItem { ident, .. }) => Some(*ident),
            OwnerNode::Crate(..) => None,
        }
    }

    pub fn span(&self) -> Span {
        match self {
            OwnerNode::Item(Item { span, .. })
            | OwnerNode::ForeignItem(ForeignItem { span, .. })
            | OwnerNode::ImplItem(ImplItem { span, .. })
            | OwnerNode::TraitItem(TraitItem { span, .. })
            | OwnerNode::Crate(Mod { inner: span, .. }) => *span,
        }
    }

    pub fn fn_decl(&self) -> Option<&FnDecl<'hir>> {
        match self {
            OwnerNode::TraitItem(TraitItem { kind: TraitItemKind::Fn(fn_sig, _), .. })
            | OwnerNode::ImplItem(ImplItem { kind: ImplItemKind::Fn(fn_sig, _), .. })
            | OwnerNode::Item(Item { kind: ItemKind::Fn(fn_sig, _, _), .. }) => Some(fn_sig.decl),
            OwnerNode::ForeignItem(ForeignItem {
                kind: ForeignItemKind::Fn(fn_decl, _, _),
                ..
            }) => Some(fn_decl),
            _ => None,
        }
    }

    pub fn body_id(&self) -> Option<BodyId> {
        match self {
            OwnerNode::TraitItem(TraitItem {
                kind: TraitItemKind::Fn(_, TraitFn::Provided(body_id)),
                ..
            })
            | OwnerNode::ImplItem(ImplItem { kind: ImplItemKind::Fn(_, body_id), .. })
            | OwnerNode::Item(Item { kind: ItemKind::Fn(.., body_id), .. }) => Some(*body_id),
            _ => None,
        }
    }

    pub fn generics(&self) -> Option<&'hir Generics<'hir>> {
        match self {
            OwnerNode::TraitItem(TraitItem { generics, .. })
            | OwnerNode::ImplItem(ImplItem { generics, .. }) => Some(generics),
            OwnerNode::Item(item) => item.kind.generics(),
            _ => None,
        }
    }

    pub fn def_id(self) -> LocalDefId {
        match self {
            OwnerNode::Item(Item { def_id, .. })
            | OwnerNode::TraitItem(TraitItem { def_id, .. })
            | OwnerNode::ImplItem(ImplItem { def_id, .. })
            | OwnerNode::ForeignItem(ForeignItem { def_id, .. }) => *def_id,
            OwnerNode::Crate(..) => crate::CRATE_HIR_ID.owner,
        }
    }

    pub fn expect_item(self) -> &'hir Item<'hir> {
        match self {
            OwnerNode::Item(n) => n,
            _ => panic!(),
        }
    }

    pub fn expect_foreign_item(self) -> &'hir ForeignItem<'hir> {
        match self {
            OwnerNode::ForeignItem(n) => n,
            _ => panic!(),
        }
    }

    pub fn expect_impl_item(self) -> &'hir ImplItem<'hir> {
        match self {
            OwnerNode::ImplItem(n) => n,
            _ => panic!(),
        }
    }

    pub fn expect_trait_item(self) -> &'hir TraitItem<'hir> {
        match self {
            OwnerNode::TraitItem(n) => n,
            _ => panic!(),
        }
    }
}

impl<'hir> Into<OwnerNode<'hir>> for &'hir Item<'hir> {
    fn into(self) -> OwnerNode<'hir> {
        OwnerNode::Item(self)
    }
}

impl<'hir> Into<OwnerNode<'hir>> for &'hir ForeignItem<'hir> {
    fn into(self) -> OwnerNode<'hir> {
        OwnerNode::ForeignItem(self)
    }
}

impl<'hir> Into<OwnerNode<'hir>> for &'hir ImplItem<'hir> {
    fn into(self) -> OwnerNode<'hir> {
        OwnerNode::ImplItem(self)
    }
}

impl<'hir> Into<OwnerNode<'hir>> for &'hir TraitItem<'hir> {
    fn into(self) -> OwnerNode<'hir> {
        OwnerNode::TraitItem(self)
    }
}

impl<'hir> Into<Node<'hir>> for OwnerNode<'hir> {
    fn into(self) -> Node<'hir> {
        match self {
            OwnerNode::Item(n) => Node::Item(n),
            OwnerNode::ForeignItem(n) => Node::ForeignItem(n),
            OwnerNode::ImplItem(n) => Node::ImplItem(n),
            OwnerNode::TraitItem(n) => Node::TraitItem(n),
            OwnerNode::Crate(n) => Node::Crate(n),
        }
    }
}

#[derive(Copy, Clone, Debug, HashStable_Generic)]
pub enum Node<'hir> {
    Param(&'hir Param<'hir>),
    Item(&'hir Item<'hir>),
    ForeignItem(&'hir ForeignItem<'hir>),
    TraitItem(&'hir TraitItem<'hir>),
    ImplItem(&'hir ImplItem<'hir>),
    Variant(&'hir Variant<'hir>),
    Field(&'hir FieldDef<'hir>),
    AnonConst(&'hir AnonConst),
    Expr(&'hir Expr<'hir>),
    Stmt(&'hir Stmt<'hir>),
    PathSegment(&'hir PathSegment<'hir>),
    Ty(&'hir Ty<'hir>),
    TraitRef(&'hir TraitRef<'hir>),
    Binding(&'hir Pat<'hir>),
    Pat(&'hir Pat<'hir>),
    Arm(&'hir Arm<'hir>),
    Block(&'hir Block<'hir>),
    Local(&'hir Local<'hir>),

    /// `Ctor` refers to the constructor of an enum variant or struct. Only tuple or unit variants
    /// with synthesized constructors.
    Ctor(&'hir VariantData<'hir>),

    Lifetime(&'hir Lifetime),
    GenericParam(&'hir GenericParam<'hir>),
    Visibility(&'hir Visibility<'hir>),

    Crate(&'hir Mod<'hir>),

    Infer(&'hir InferArg),
}

impl<'hir> Node<'hir> {
    /// Get the identifier of this `Node`, if applicable.
    ///
    /// # Edge cases
    ///
    /// Calling `.ident()` on a [`Node::Ctor`] will return `None`
    /// because `Ctor`s do not have identifiers themselves.
    /// Instead, call `.ident()` on the parent struct/variant, like so:
    ///
    /// ```ignore (illustrative)
    /// ctor
    ///     .ctor_hir_id()
    ///     .and_then(|ctor_id| tcx.hir().find(tcx.hir().get_parent_node(ctor_id)))
    ///     .and_then(|parent| parent.ident())
    /// ```
    pub fn ident(&self) -> Option<Ident> {
        match self {
            Node::TraitItem(TraitItem { ident, .. })
            | Node::ImplItem(ImplItem { ident, .. })
            | Node::ForeignItem(ForeignItem { ident, .. })
            | Node::Field(FieldDef { ident, .. })
            | Node::Variant(Variant { ident, .. })
            | Node::Item(Item { ident, .. })
            | Node::PathSegment(PathSegment { ident, .. }) => Some(*ident),
            Node::Lifetime(lt) => Some(lt.name.ident()),
            Node::GenericParam(p) => Some(p.name.ident()),
            Node::Param(..)
            | Node::AnonConst(..)
            | Node::Expr(..)
            | Node::Stmt(..)
            | Node::Block(..)
            | Node::Ctor(..)
            | Node::Pat(..)
            | Node::Binding(..)
            | Node::Arm(..)
            | Node::Local(..)
            | Node::Visibility(..)
            | Node::Crate(..)
            | Node::Ty(..)
            | Node::TraitRef(..)
            | Node::Infer(..) => None,
        }
    }

    pub fn fn_decl(&self) -> Option<&FnDecl<'hir>> {
        match self {
            Node::TraitItem(TraitItem { kind: TraitItemKind::Fn(fn_sig, _), .. })
            | Node::ImplItem(ImplItem { kind: ImplItemKind::Fn(fn_sig, _), .. })
            | Node::Item(Item { kind: ItemKind::Fn(fn_sig, _, _), .. }) => Some(fn_sig.decl),
            Node::ForeignItem(ForeignItem { kind: ForeignItemKind::Fn(fn_decl, _, _), .. }) => {
                Some(fn_decl)
            }
            _ => None,
        }
    }

    pub fn body_id(&self) -> Option<BodyId> {
        match self {
            Node::TraitItem(TraitItem {
                kind: TraitItemKind::Fn(_, TraitFn::Provided(body_id)),
                ..
            })
            | Node::ImplItem(ImplItem { kind: ImplItemKind::Fn(_, body_id), .. })
            | Node::Item(Item { kind: ItemKind::Fn(.., body_id), .. }) => Some(*body_id),
            _ => None,
        }
    }

    pub fn generics(&self) -> Option<&'hir Generics<'hir>> {
        match self {
            Node::TraitItem(TraitItem { generics, .. })
            | Node::ImplItem(ImplItem { generics, .. }) => Some(generics),
            Node::Item(item) => item.kind.generics(),
            _ => None,
        }
    }

    pub fn hir_id(&self) -> Option<HirId> {
        match self {
            Node::Item(Item { def_id, .. })
            | Node::TraitItem(TraitItem { def_id, .. })
            | Node::ImplItem(ImplItem { def_id, .. })
            | Node::ForeignItem(ForeignItem { def_id, .. }) => Some(HirId::make_owner(*def_id)),
            Node::Field(FieldDef { hir_id, .. })
            | Node::AnonConst(AnonConst { hir_id, .. })
            | Node::Expr(Expr { hir_id, .. })
            | Node::Stmt(Stmt { hir_id, .. })
            | Node::Ty(Ty { hir_id, .. })
            | Node::Binding(Pat { hir_id, .. })
            | Node::Pat(Pat { hir_id, .. })
            | Node::Arm(Arm { hir_id, .. })
            | Node::Block(Block { hir_id, .. })
            | Node::Local(Local { hir_id, .. })
            | Node::Lifetime(Lifetime { hir_id, .. })
            | Node::Param(Param { hir_id, .. })
            | Node::Infer(InferArg { hir_id, .. })
            | Node::GenericParam(GenericParam { hir_id, .. }) => Some(*hir_id),
            Node::TraitRef(TraitRef { hir_ref_id, .. }) => Some(*hir_ref_id),
            Node::PathSegment(PathSegment { hir_id, .. }) => *hir_id,
            Node::Variant(Variant { id, .. }) => Some(*id),
            Node::Ctor(variant) => variant.ctor_hir_id(),
            Node::Crate(_) | Node::Visibility(_) => None,
        }
    }

    /// Returns `Constness::Const` when this node is a const fn/impl/item.
    pub fn constness_for_typeck(&self) -> Constness {
        match self {
            Node::Item(Item {
                kind: ItemKind::Fn(FnSig { header: FnHeader { constness, .. }, .. }, ..),
                ..
            })
            | Node::TraitItem(TraitItem {
                kind: TraitItemKind::Fn(FnSig { header: FnHeader { constness, .. }, .. }, ..),
                ..
            })
            | Node::ImplItem(ImplItem {
                kind: ImplItemKind::Fn(FnSig { header: FnHeader { constness, .. }, .. }, ..),
                ..
            })
            | Node::Item(Item { kind: ItemKind::Impl(Impl { constness, .. }), .. }) => *constness,

            Node::Item(Item { kind: ItemKind::Const(..), .. })
            | Node::TraitItem(TraitItem { kind: TraitItemKind::Const(..), .. })
            | Node::ImplItem(ImplItem { kind: ImplItemKind::Const(..), .. }) => Constness::Const,

            _ => Constness::NotConst,
        }
    }

    pub fn as_owner(self) -> Option<OwnerNode<'hir>> {
        match self {
            Node::Item(i) => Some(OwnerNode::Item(i)),
            Node::ForeignItem(i) => Some(OwnerNode::ForeignItem(i)),
            Node::TraitItem(i) => Some(OwnerNode::TraitItem(i)),
            Node::ImplItem(i) => Some(OwnerNode::ImplItem(i)),
            Node::Crate(i) => Some(OwnerNode::Crate(i)),
            _ => None,
        }
    }
}

// Some nodes are used a lot. Make sure they don't unintentionally get bigger.
#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
mod size_asserts {
    rustc_data_structures::static_assert_size!(super::Block<'static>, 48);
    rustc_data_structures::static_assert_size!(super::Expr<'static>, 64);
    rustc_data_structures::static_assert_size!(super::Pat<'static>, 88);
    rustc_data_structures::static_assert_size!(super::QPath<'static>, 24);
    rustc_data_structures::static_assert_size!(super::Ty<'static>, 72);

    rustc_data_structures::static_assert_size!(super::Item<'static>, 184);
    rustc_data_structures::static_assert_size!(super::TraitItem<'static>, 128);
    rustc_data_structures::static_assert_size!(super::ImplItem<'static>, 152);
    rustc_data_structures::static_assert_size!(super::ForeignItem<'static>, 136);
}
