//! HIR datatypes. See the [rustc guide] for more info.
//!
//! [rustc guide]: https://rust-lang.github.io/rustc-guide/hir.html

pub use self::BlockCheckMode::*;
pub use self::CaptureClause::*;
pub use self::FunctionRetTy::*;
pub use self::Mutability::*;
pub use self::PrimTy::*;
pub use self::UnOp::*;
pub use self::UnsafeSource::*;

use crate::hir::def::{Res, DefKind};
use crate::hir::def_id::{DefId, DefIndex, LocalDefId, CRATE_DEF_INDEX};
use crate::hir::ptr::P;
use crate::util::nodemap::{NodeMap, FxHashSet};
use crate::mir::mono::Linkage;

use errors::FatalError;
use syntax_pos::{Span, DUMMY_SP, symbol::InternedString, MultiSpan};
use syntax::source_map::Spanned;
use rustc_target::spec::abi::Abi;
use syntax::ast::{self, CrateSugar, Ident, Name, NodeId, AsmDialect};
use syntax::ast::{Attribute, Label, LitKind, StrStyle, FloatTy, IntTy, UintTy};
use syntax::attr::{InlineAttr, OptimizeAttr};
use syntax::ext::hygiene::SyntaxContext;
use syntax::symbol::{Symbol, kw};
use syntax::tokenstream::TokenStream;
use syntax::util::parser::ExprPrecedence;
use crate::ty::AdtKind;
use crate::ty::query::Providers;

use rustc_data_structures::sync::{par_for_each_in, Send, Sync};
use rustc_data_structures::thin_vec::ThinVec;
use rustc_macros::HashStable;

use serialize::{self, Encoder, Encodable, Decoder, Decodable};
use std::collections::{BTreeSet, BTreeMap};
use std::fmt;
use smallvec::SmallVec;

/// HIR doesn't commit to a concrete storage type and has its own alias for a vector.
/// It can be `Vec`, `P<[T]>` or potentially `Box<[T]>`, or some other container with similar
/// behavior. Unlike AST, HIR is mostly a static structure, so we can use an owned slice instead
/// of `Vec` to avoid keeping extra capacity.
pub type HirVec<T> = P<[T]>;

macro_rules! hir_vec {
    ($elem:expr; $n:expr) => (
        $crate::hir::HirVec::from(vec![$elem; $n])
    );
    ($($x:expr),*) => (
        $crate::hir::HirVec::from(vec![$($x),*])
    );
}

pub mod check_attr;
pub mod def;
pub mod def_id;
pub mod intravisit;
pub mod itemlikevisit;
pub mod lowering;
pub mod map;
pub mod pat_util;
pub mod print;
pub mod ptr;
pub mod upvars;

/// Uniquely identifies a node in the HIR of the current crate. It is
/// composed of the `owner`, which is the `DefIndex` of the directly enclosing
/// `hir::Item`, `hir::TraitItem`, or `hir::ImplItem` (i.e., the closest "item-like"),
/// and the `local_id` which is unique within the given owner.
///
/// This two-level structure makes for more stable values: One can move an item
/// around within the source code, or add or remove stuff before it, without
/// the `local_id` part of the `HirId` changing, which is a very useful property in
/// incremental compilation where we have to persist things through changes to
/// the code base.
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
pub struct HirId {
    pub owner: DefIndex,
    pub local_id: ItemLocalId,
}

impl HirId {
    pub fn owner_def_id(self) -> DefId {
        DefId::local(self.owner)
    }

    pub fn owner_local_def_id(self) -> LocalDefId {
        LocalDefId::from_def_id(DefId::local(self.owner))
    }
}

impl serialize::UseSpecializedEncodable for HirId {
    fn default_encode<S: Encoder>(&self, s: &mut S) -> Result<(), S::Error> {
        let HirId {
            owner,
            local_id,
        } = *self;

        owner.encode(s)?;
        local_id.encode(s)
    }
}

impl serialize::UseSpecializedDecodable for HirId {
    fn default_decode<D: Decoder>(d: &mut D) -> Result<HirId, D::Error> {
        let owner = DefIndex::decode(d)?;
        let local_id = ItemLocalId::decode(d)?;

        Ok(HirId {
            owner,
            local_id
        })
    }
}

impl fmt::Display for HirId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

// Hack to ensure that we don't try to access the private parts of `ItemLocalId` in this module
mod item_local_id_inner {
    use rustc_data_structures::indexed_vec::Idx;
    use rustc_macros::HashStable;
    newtype_index! {
        /// An `ItemLocalId` uniquely identifies something within a given "item-like";
        /// that is, within a `hir::Item`, `hir::TraitItem`, or `hir::ImplItem`. There is no
        /// guarantee that the numerical value of a given `ItemLocalId` corresponds to
        /// the node's position within the owning item in any way, but there is a
        /// guarantee that the `LocalItemId`s within an owner occupy a dense range of
        /// integers starting at zero, so a mapping that maps all or most nodes within
        /// an "item-like" to something else can be implemented by a `Vec` instead of a
        /// tree or hash map.
        pub struct ItemLocalId {
            derive [HashStable]
        }
    }
}

pub use self::item_local_id_inner::ItemLocalId;

/// The `HirId` corresponding to `CRATE_NODE_ID` and `CRATE_DEF_INDEX`.
pub const CRATE_HIR_ID: HirId = HirId {
    owner: CRATE_DEF_INDEX,
    local_id: ItemLocalId::from_u32_const(0)
};

pub const DUMMY_HIR_ID: HirId = HirId {
    owner: CRATE_DEF_INDEX,
    local_id: DUMMY_ITEM_LOCAL_ID,
};

pub const DUMMY_ITEM_LOCAL_ID: ItemLocalId = ItemLocalId::MAX;

#[derive(Copy, Clone, RustcEncodable, RustcDecodable, HashStable)]
pub struct Lifetime {
    pub hir_id: HirId,
    pub span: Span,

    /// Either "`'a`", referring to a named lifetime definition,
    /// or "``" (i.e., `kw::Invalid`), for elision placeholders.
    ///
    /// HIR lowering inserts these placeholders in type paths that
    /// refer to type definitions needing lifetime parameters,
    /// `&T` and `&mut T`, and trait objects without `... + 'a`.
    pub name: LifetimeName,
}

#[derive(Debug, Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Copy, HashStable)]
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
    /// repored (so we should squelch other derived errors). Occurs
    /// when, e.g., `'_` is used in the wrong place.
    Error,
}

impl ParamName {
    pub fn ident(&self) -> Ident {
        match *self {
            ParamName::Plain(ident) => ident,
            ParamName::Fresh(_) |
            ParamName::Error => Ident::with_empty_ctxt(kw::UnderscoreLifetime),
        }
    }

    pub fn modern(&self) -> ParamName {
        match *self {
            ParamName::Plain(ident) => ParamName::Plain(ident.modern()),
            param_name => param_name,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Copy, HashStable)]
pub enum LifetimeName {
    /// User-given names or fresh (synthetic) names.
    Param(ParamName),

    /// User wrote nothing (e.g., the lifetime in `&u32`).
    Implicit,

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
            LifetimeName::Implicit | LifetimeName::Error => Ident::invalid(),
            LifetimeName::Underscore => Ident::with_empty_ctxt(kw::UnderscoreLifetime),
            LifetimeName::Static => Ident::with_empty_ctxt(kw::StaticLifetime),
            LifetimeName::Param(param_name) => param_name.ident(),
        }
    }

    pub fn is_elided(&self) -> bool {
        match self {
            LifetimeName::Implicit | LifetimeName::Underscore => true,

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

    pub fn modern(&self) -> LifetimeName {
        match *self {
            LifetimeName::Param(param_name) => LifetimeName::Param(param_name.modern()),
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
        write!(f,
               "lifetime({}: {})",
               self.hir_id,
               print::to_string(print::NO_ANN, |s| s.print_lifetime(self)))
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
#[derive(RustcEncodable, RustcDecodable, HashStable)]
pub struct Path {
    pub span: Span,
    /// The resolution for the path.
    pub res: Res,
    /// The segments in the path: the things separated by `::`.
    pub segments: HirVec<PathSegment>,
}

impl Path {
    pub fn is_global(&self) -> bool {
        !self.segments.is_empty() && self.segments[0].ident.name == kw::PathRoot
    }
}

impl fmt::Debug for Path {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "path({})", self)
    }
}

impl fmt::Display for Path {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", print::to_string(print::NO_ANN, |s| s.print_path(self, false)))
    }
}

/// A segment of a path: an identifier, an optional lifetime, and a set of
/// types.
#[derive(RustcEncodable, RustcDecodable, Debug, HashStable)]
pub struct PathSegment {
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
    pub args: Option<P<GenericArgs>>,

    /// Whether to infer remaining type parameters, if any.
    /// This only applies to expression and pattern paths, and
    /// out of those only the segments with no type parameters
    /// to begin with, e.g., `Vec::new` is `<Vec<..>>::new::<..>`.
    pub infer_args: bool,
}

impl PathSegment {
    /// Converts an identifier to the corresponding segment.
    pub fn from_ident(ident: Ident) -> PathSegment {
        PathSegment {
            ident,
            hir_id: None,
            res: None,
            infer_args: true,
            args: None,
        }
    }

    pub fn new(
        ident: Ident,
        hir_id: Option<HirId>,
        res: Option<Res>,
        args: GenericArgs,
        infer_args: bool,
    ) -> Self {
        PathSegment {
            ident,
            hir_id,
            res,
            infer_args,
            args: if args.is_empty() {
                None
            } else {
                Some(P(args))
            }
        }
    }

    pub fn generic_args(&self) -> &GenericArgs {
        if let Some(ref args) = self.args {
            args
        } else {
            const DUMMY: &GenericArgs = &GenericArgs::none();
            DUMMY
        }
    }
}

#[derive(RustcEncodable, RustcDecodable, Debug, HashStable)]
pub struct ConstArg {
    pub value: AnonConst,
    pub span: Span,
}

#[derive(RustcEncodable, RustcDecodable, Debug, HashStable)]
pub enum GenericArg {
    Lifetime(Lifetime),
    Type(Ty),
    Const(ConstArg),
}

impl GenericArg {
    pub fn span(&self) -> Span {
        match self {
            GenericArg::Lifetime(l) => l.span,
            GenericArg::Type(t) => t.span,
            GenericArg::Const(c) => c.span,
        }
    }

    pub fn id(&self) -> HirId {
        match self {
            GenericArg::Lifetime(l) => l.hir_id,
            GenericArg::Type(t) => t.hir_id,
            GenericArg::Const(c) => c.value.hir_id,
        }
    }

    pub fn is_const(&self) -> bool {
        match self {
            GenericArg::Const(_) => true,
            _ => false,
        }
    }
}

#[derive(RustcEncodable, RustcDecodable, Debug, HashStable)]
pub struct GenericArgs {
    /// The generic arguments for this path segment.
    pub args: HirVec<GenericArg>,
    /// Bindings (equality constraints) on associated types, if present.
    /// E.g., `Foo<A = Bar>`.
    pub bindings: HirVec<TypeBinding>,
    /// Were arguments written in parenthesized form `Fn(T) -> U`?
    /// This is required mostly for pretty-printing and diagnostics,
    /// but also for changing lifetime elision rules to be "function-like".
    pub parenthesized: bool,
}

impl GenericArgs {
    pub const fn none() -> Self {
        Self {
            args: HirVec::new(),
            bindings: HirVec::new(),
            parenthesized: false,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.args.is_empty() && self.bindings.is_empty() && !self.parenthesized
    }

    pub fn inputs(&self) -> &[Ty] {
        if self.parenthesized {
            for arg in &self.args {
                match arg {
                    GenericArg::Lifetime(_) => {}
                    GenericArg::Type(ref ty) => {
                        if let TyKind::Tup(ref tys) = ty.node {
                            return tys;
                        }
                        break;
                    }
                    GenericArg::Const(_) => {}
                }
            }
        }
        bug!("GenericArgs::inputs: not a `Fn(T) -> U`");
    }

    pub fn own_counts(&self) -> GenericParamCount {
        // We could cache this as a property of `GenericParamCount`, but
        // the aim is to refactor this away entirely eventually and the
        // presence of this method will be a constant reminder.
        let mut own_counts: GenericParamCount = Default::default();

        for arg in &self.args {
            match arg {
                GenericArg::Lifetime(_) => own_counts.lifetimes += 1,
                GenericArg::Type(_) => own_counts.types += 1,
                GenericArg::Const(_) => own_counts.consts += 1,
            };
        }

        own_counts
    }
}

/// A modifier on a bound, currently this is only used for `?Sized`, where the
/// modifier is `Maybe`. Negative bounds should also be handled here.
#[derive(Copy, Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Debug, HashStable)]
pub enum TraitBoundModifier {
    None,
    Maybe,
}

/// The AST represents all type param bounds as types.
/// `typeck::collect::compute_bounds` matches these against
/// the "special" built-in traits (see `middle::lang_items`) and
/// detects `Copy`, `Send` and `Sync`.
#[derive(RustcEncodable, RustcDecodable, Debug, HashStable)]
pub enum GenericBound {
    Trait(PolyTraitRef, TraitBoundModifier),
    Outlives(Lifetime),
}

impl GenericBound {
    pub fn span(&self) -> Span {
        match self {
            &GenericBound::Trait(ref t, ..) => t.span,
            &GenericBound::Outlives(ref l) => l.span,
        }
    }
}

pub type GenericBounds = HirVec<GenericBound>;

#[derive(Copy, Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Debug, HashStable)]
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

#[derive(RustcEncodable, RustcDecodable, Debug, HashStable)]
pub enum GenericParamKind {
    /// A lifetime definition (e.g., `'a: 'b + 'c + 'd`).
    Lifetime {
        kind: LifetimeParamKind,
    },
    Type {
        default: Option<P<Ty>>,
        synthetic: Option<SyntheticTyParamKind>,
    },
    Const {
        ty: P<Ty>,
    }
}

#[derive(RustcEncodable, RustcDecodable, Debug, HashStable)]
pub struct GenericParam {
    pub hir_id: HirId,
    pub name: ParamName,
    pub attrs: HirVec<Attribute>,
    pub bounds: GenericBounds,
    pub span: Span,
    pub pure_wrt_drop: bool,
    pub kind: GenericParamKind,
}

#[derive(Default)]
pub struct GenericParamCount {
    pub lifetimes: usize,
    pub types: usize,
    pub consts: usize,
}

/// Represents lifetimes and type parameters attached to a declaration
/// of a function, enum, trait, etc.
#[derive(RustcEncodable, RustcDecodable, Debug, HashStable)]
pub struct Generics {
    pub params: HirVec<GenericParam>,
    pub where_clause: WhereClause,
    pub span: Span,
}

impl Generics {
    pub const fn empty() -> Generics {
        Generics {
            params: HirVec::new(),
            where_clause: WhereClause {
                predicates: HirVec::new(),
                span: DUMMY_SP,
            },
            span: DUMMY_SP,
        }
    }

    pub fn own_counts(&self) -> GenericParamCount {
        // We could cache this as a property of `GenericParamCount`, but
        // the aim is to refactor this away entirely eventually and the
        // presence of this method will be a constant reminder.
        let mut own_counts: GenericParamCount = Default::default();

        for param in &self.params {
            match param.kind {
                GenericParamKind::Lifetime { .. } => own_counts.lifetimes += 1,
                GenericParamKind::Type { .. } => own_counts.types += 1,
                GenericParamKind::Const { .. } => own_counts.consts += 1,
            };
        }

        own_counts
    }

    pub fn get_named(&self, name: InternedString) -> Option<&GenericParam> {
        for param in &self.params {
            if name == param.name.ident().as_interned_str() {
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
#[derive(Copy, Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Debug, HashStable)]
pub enum SyntheticTyParamKind {
    ImplTrait
}

/// A where-clause in a definition.
#[derive(RustcEncodable, RustcDecodable, Debug, HashStable)]
pub struct WhereClause {
    pub predicates: HirVec<WherePredicate>,
    // Only valid if predicates isn't empty.
    span: Span,
}

impl WhereClause {
    pub fn span(&self) -> Option<Span> {
        if self.predicates.is_empty() {
            None
        } else {
            Some(self.span)
        }
    }
}

/// A single predicate in a where-clause.
#[derive(RustcEncodable, RustcDecodable, Debug, HashStable)]
pub enum WherePredicate {
    /// A type binding (e.g., `for<'c> Foo: Send + Clone + 'c`).
    BoundPredicate(WhereBoundPredicate),
    /// A lifetime predicate (e.g., `'a: 'b + 'c`).
    RegionPredicate(WhereRegionPredicate),
    /// An equality predicate (unsupported).
    EqPredicate(WhereEqPredicate),
}

impl WherePredicate {
    pub fn span(&self) -> Span {
        match self {
            &WherePredicate::BoundPredicate(ref p) => p.span,
            &WherePredicate::RegionPredicate(ref p) => p.span,
            &WherePredicate::EqPredicate(ref p) => p.span,
        }
    }
}

/// A type bound (e.g., `for<'c> Foo: Send + Clone + 'c`).
#[derive(RustcEncodable, RustcDecodable, Debug, HashStable)]
pub struct WhereBoundPredicate {
    pub span: Span,
    /// Any generics from a `for` binding.
    pub bound_generic_params: HirVec<GenericParam>,
    /// The type being bounded.
    pub bounded_ty: P<Ty>,
    /// Trait and lifetime bounds (e.g., `Clone + Send + 'static`).
    pub bounds: GenericBounds,
}

/// A lifetime predicate (e.g., `'a: 'b + 'c`).
#[derive(RustcEncodable, RustcDecodable, Debug, HashStable)]
pub struct WhereRegionPredicate {
    pub span: Span,
    pub lifetime: Lifetime,
    pub bounds: GenericBounds,
}

/// An equality predicate (e.g., `T = int`); currently unsupported.
#[derive(RustcEncodable, RustcDecodable, Debug, HashStable)]
pub struct WhereEqPredicate {
    pub hir_id: HirId,
    pub span: Span,
    pub lhs_ty: P<Ty>,
    pub rhs_ty: P<Ty>,
}

#[derive(RustcEncodable, RustcDecodable, Debug)]
pub struct ModuleItems {
    // Use BTreeSets here so items are in the same order as in the
    // list of all items in Crate
    pub items: BTreeSet<HirId>,
    pub trait_items: BTreeSet<TraitItemId>,
    pub impl_items: BTreeSet<ImplItemId>,
}

/// The top-level data structure that stores the entire contents of
/// the crate currently being compiled.
///
/// For more details, see the [rustc guide].
///
/// [rustc guide]: https://rust-lang.github.io/rustc-guide/hir.html
#[derive(RustcEncodable, RustcDecodable, Debug)]
pub struct Crate {
    pub module: Mod,
    pub attrs: HirVec<Attribute>,
    pub span: Span,
    pub exported_macros: HirVec<MacroDef>,

    // N.B., we use a BTreeMap here so that `visit_all_items` iterates
    // over the ids in increasing order. In principle it should not
    // matter what order we visit things in, but in *practice* it
    // does, because it can affect the order in which errors are
    // detected, which in turn can make compile-fail tests yield
    // slightly different results.
    pub items: BTreeMap<HirId, Item>,

    pub trait_items: BTreeMap<TraitItemId, TraitItem>,
    pub impl_items: BTreeMap<ImplItemId, ImplItem>,
    pub bodies: BTreeMap<BodyId, Body>,
    pub trait_impls: BTreeMap<DefId, Vec<HirId>>,

    /// A list of the body ids written out in the order in which they
    /// appear in the crate. If you're going to process all the bodies
    /// in the crate, you should iterate over this list rather than the keys
    /// of bodies.
    pub body_ids: Vec<BodyId>,

    /// A list of modules written out in the order in which they
    /// appear in the crate. This includes the main crate module.
    pub modules: BTreeMap<NodeId, ModuleItems>,
}

impl Crate {
    pub fn item(&self, id: HirId) -> &Item {
        &self.items[&id]
    }

    pub fn trait_item(&self, id: TraitItemId) -> &TraitItem {
        &self.trait_items[&id]
    }

    pub fn impl_item(&self, id: ImplItemId) -> &ImplItem {
        &self.impl_items[&id]
    }

    /// Visits all items in the crate in some deterministic (but
    /// unspecified) order. If you just need to process every item,
    /// but don't care about nesting, this method is the best choice.
    ///
    /// If you do care about nesting -- usually because your algorithm
    /// follows lexical scoping rules -- then you want a different
    /// approach. You should override `visit_nested_item` in your
    /// visitor and then call `intravisit::walk_crate` instead.
    pub fn visit_all_item_likes<'hir, V>(&'hir self, visitor: &mut V)
        where V: itemlikevisit::ItemLikeVisitor<'hir>
    {
        for (_, item) in &self.items {
            visitor.visit_item(item);
        }

        for (_, trait_item) in &self.trait_items {
            visitor.visit_trait_item(trait_item);
        }

        for (_, impl_item) in &self.impl_items {
            visitor.visit_impl_item(impl_item);
        }
    }

    /// A parallel version of `visit_all_item_likes`.
    pub fn par_visit_all_item_likes<'hir, V>(&'hir self, visitor: &V)
        where V: itemlikevisit::ParItemLikeVisitor<'hir> + Sync + Send
    {
        parallel!({
            par_for_each_in(&self.items, |(_, item)| {
                visitor.visit_item(item);
            });
        }, {
            par_for_each_in(&self.trait_items, |(_, trait_item)| {
                visitor.visit_trait_item(trait_item);
            });
        }, {
            par_for_each_in(&self.impl_items, |(_, impl_item)| {
                visitor.visit_impl_item(impl_item);
            });
        });
    }

    pub fn body(&self, id: BodyId) -> &Body {
        &self.bodies[&id]
    }
}

/// A macro definition, in this crate or imported from another.
///
/// Not parsed directly, but created on macro import or `macro_rules!` expansion.
#[derive(RustcEncodable, RustcDecodable, Debug, HashStable)]
pub struct MacroDef {
    pub name: Name,
    pub vis: Visibility,
    pub attrs: HirVec<Attribute>,
    pub hir_id: HirId,
    pub span: Span,
    pub body: TokenStream,
    pub legacy: bool,
}

/// A block of statements `{ .. }`, which may have a label (in this case the
/// `targeted_by_break` field will be `true`) and may be `unsafe` by means of
/// the `rules` being anything but `DefaultBlock`.
#[derive(RustcEncodable, RustcDecodable, Debug, HashStable)]
pub struct Block {
    /// Statements in a block.
    pub stmts: HirVec<Stmt>,
    /// An expression at the end of the block
    /// without a semicolon, if any.
    pub expr: Option<P<Expr>>,
    #[stable_hasher(ignore)]
    pub hir_id: HirId,
    /// Distinguishes between `unsafe { ... }` and `{ ... }`.
    pub rules: BlockCheckMode,
    pub span: Span,
    /// If true, then there may exist `break 'a` values that aim to
    /// break out of this block early.
    /// Used by `'label: {}` blocks and by `catch` statements.
    pub targeted_by_break: bool,
}

#[derive(RustcEncodable, RustcDecodable, HashStable)]
pub struct Pat {
    #[stable_hasher(ignore)]
    pub hir_id: HirId,
    pub node: PatKind,
    pub span: Span,
}

impl fmt::Debug for Pat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "pat({}: {})", self.hir_id,
               print::to_string(print::NO_ANN, |s| s.print_pat(self)))
    }
}

impl Pat {
    // FIXME(#19596) this is a workaround, but there should be a better way
    fn walk_<G>(&self, it: &mut G) -> bool
        where G: FnMut(&Pat) -> bool
    {
        if !it(self) {
            return false;
        }

        match self.node {
            PatKind::Binding(.., Some(ref p)) => p.walk_(it),
            PatKind::Struct(_, ref fields, _) => {
                fields.iter().all(|field| field.node.pat.walk_(it))
            }
            PatKind::TupleStruct(_, ref s, _) | PatKind::Tuple(ref s, _) => {
                s.iter().all(|p| p.walk_(it))
            }
            PatKind::Box(ref s) | PatKind::Ref(ref s, _) => {
                s.walk_(it)
            }
            PatKind::Slice(ref before, ref slice, ref after) => {
                before.iter()
                      .chain(slice.iter())
                      .chain(after.iter())
                      .all(|p| p.walk_(it))
            }
            PatKind::Wild |
            PatKind::Lit(_) |
            PatKind::Range(..) |
            PatKind::Binding(..) |
            PatKind::Path(_) => {
                true
            }
        }
    }

    pub fn walk<F>(&self, mut it: F) -> bool
        where F: FnMut(&Pat) -> bool
    {
        self.walk_(&mut it)
    }
}

/// A single field in a struct pattern.
///
/// Patterns like the fields of Foo `{ x, ref y, ref mut z }`
/// are treated the same as` x: x, y: ref y, z: ref mut z`,
/// except `is_shorthand` is true.
#[derive(RustcEncodable, RustcDecodable, Debug, HashStable)]
pub struct FieldPat {
    #[stable_hasher(ignore)]
    pub hir_id: HirId,
    /// The identifier for the field.
    #[stable_hasher(project(name))]
    pub ident: Ident,
    /// The pattern the field is destructured to.
    pub pat: P<Pat>,
    pub is_shorthand: bool,
}

/// Explicit binding annotations given in the HIR for a binding. Note
/// that this is not the final binding *mode* that we infer after type
/// inference.
#[derive(Copy, Clone, PartialEq, RustcEncodable, RustcDecodable, Debug, HashStable)]
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

#[derive(Copy, Clone, PartialEq, RustcEncodable, RustcDecodable, Debug, HashStable)]
pub enum RangeEnd {
    Included,
    Excluded,
}

#[derive(RustcEncodable, RustcDecodable, Debug, HashStable)]
pub enum PatKind {
    /// Represents a wildcard pattern (i.e., `_`).
    Wild,

    /// A fresh binding `ref mut binding @ OPT_SUBPATTERN`.
    /// The `HirId` is the canonical ID for the variable being bound,
    /// (e.g., in `Ok(x) | Err(x)`, both `x` use the same canonical ID),
    /// which is the pattern ID of the first `x`.
    Binding(BindingAnnotation, HirId, Ident, Option<P<Pat>>),

    /// A struct or struct variant pattern (e.g., `Variant {x, y, ..}`).
    /// The `bool` is `true` in the presence of a `..`.
    Struct(QPath, HirVec<Spanned<FieldPat>>, bool),

    /// A tuple struct/variant pattern `Variant(x, y, .., z)`.
    /// If the `..` pattern fragment is present, then `Option<usize>` denotes its position.
    /// `0 <= position <= subpats.len()`
    TupleStruct(QPath, HirVec<P<Pat>>, Option<usize>),

    /// A path pattern for an unit struct/variant or a (maybe-associated) constant.
    Path(QPath),

    /// A tuple pattern (e.g., `(a, b)`).
    /// If the `..` pattern fragment is present, then `Option<usize>` denotes its position.
    /// `0 <= position <= subpats.len()`
    Tuple(HirVec<P<Pat>>, Option<usize>),

    /// A `box` pattern.
    Box(P<Pat>),

    /// A reference pattern (e.g., `&mut (a, b)`).
    Ref(P<Pat>, Mutability),

    /// A literal.
    Lit(P<Expr>),

    /// A range pattern (e.g., `1..=2` or `1..2`).
    Range(P<Expr>, P<Expr>, RangeEnd),

    /// `[a, b, ..i, y, z]` is represented as:
    ///     `PatKind::Slice(box [a, b], Some(i), box [y, z])`.
    Slice(HirVec<P<Pat>>, Option<P<Pat>>, HirVec<P<Pat>>),
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, HashStable,
         RustcEncodable, RustcDecodable, Hash, Debug)]
pub enum Mutability {
    MutMutable,
    MutImmutable,
}

impl Mutability {
    /// Returns `MutMutable` only if both arguments are mutable.
    pub fn and(self, other: Self) -> Self {
        match self {
            MutMutable => other,
            MutImmutable => MutImmutable,
        }
    }
}

#[derive(Copy, Clone, PartialEq, RustcEncodable, RustcDecodable, Debug, Hash, HashStable)]
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
        match self {
            BinOpKind::And | BinOpKind::Or => true,
            _ => false,
        }
    }

    pub fn is_shift(self) -> bool {
        match self {
            BinOpKind::Shl | BinOpKind::Shr => true,
            _ => false,
        }
    }

    pub fn is_comparison(self) -> bool {
        match self {
            BinOpKind::Eq |
            BinOpKind::Lt |
            BinOpKind::Le |
            BinOpKind::Ne |
            BinOpKind::Gt |
            BinOpKind::Ge => true,
            BinOpKind::And |
            BinOpKind::Or |
            BinOpKind::Add |
            BinOpKind::Sub |
            BinOpKind::Mul |
            BinOpKind::Div |
            BinOpKind::Rem |
            BinOpKind::BitXor |
            BinOpKind::BitAnd |
            BinOpKind::BitOr |
            BinOpKind::Shl |
            BinOpKind::Shr => false,
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

#[derive(Copy, Clone, PartialEq, RustcEncodable, RustcDecodable, Debug, Hash, HashStable)]
pub enum UnOp {
    /// The `*` operator (deferencing).
    UnDeref,
    /// The `!` operator (logical negation).
    UnNot,
    /// The `-` operator (negation).
    UnNeg,
}

impl UnOp {
    pub fn as_str(self) -> &'static str {
        match self {
            UnDeref => "*",
            UnNot => "!",
            UnNeg => "-",
        }
    }

    /// Returns `true` if the unary operator takes its argument by value.
    pub fn is_by_value(self) -> bool {
        match self {
            UnNeg | UnNot => true,
            _ => false,
        }
    }
}

/// A statement.
#[derive(RustcEncodable, RustcDecodable)]
pub struct Stmt {
    pub hir_id: HirId,
    pub node: StmtKind,
    pub span: Span,
}

impl fmt::Debug for Stmt {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "stmt({}: {})", self.hir_id,
               print::to_string(print::NO_ANN, |s| s.print_stmt(self)))
    }
}

/// The contents of a statement.
#[derive(RustcEncodable, RustcDecodable, HashStable)]
pub enum StmtKind {
    /// A local (`let`) binding.
    Local(P<Local>),

    /// An item binding.
    Item(ItemId),

    /// An expression without a trailing semi-colon (must have unit type).
    Expr(P<Expr>),

    /// An expression with a trailing semi-colon (may have any type).
    Semi(P<Expr>),
}

impl StmtKind {
    pub fn attrs(&self) -> &[Attribute] {
        match *self {
            StmtKind::Local(ref l) => &l.attrs,
            StmtKind::Item(_) => &[],
            StmtKind::Expr(ref e) |
            StmtKind::Semi(ref e) => &e.attrs,
        }
    }
}

/// Represents a `let` statement (i.e., `let <pat>:<ty> = <expr>;`).
#[derive(RustcEncodable, RustcDecodable, Debug, HashStable)]
pub struct Local {
    pub pat: P<Pat>,
    /// Type annotation, if any (otherwise the type will be inferred).
    pub ty: Option<P<Ty>>,
    /// Initializer expression to set the value, if any.
    pub init: Option<P<Expr>>,
    pub hir_id: HirId,
    pub span: Span,
    pub attrs: ThinVec<Attribute>,
    /// Can be `ForLoopDesugar` if the `let` statement is part of a `for` loop
    /// desugaring. Otherwise will be `Normal`.
    pub source: LocalSource,
}

/// Represents a single arm of a `match` expression, e.g.
/// `<pats> (if <guard>) => <body>`.
#[derive(RustcEncodable, RustcDecodable, Debug, HashStable)]
pub struct Arm {
    #[stable_hasher(ignore)]
    pub hir_id: HirId,
    pub span: Span,
    pub attrs: HirVec<Attribute>,
    /// Multiple patterns can be combined with `|`
    pub pats: HirVec<P<Pat>>,
    /// Optional guard clause.
    pub guard: Option<Guard>,
    /// The expression the arm evaluates to if this arm matches.
    pub body: P<Expr>,
}

#[derive(RustcEncodable, RustcDecodable, Debug, HashStable)]
pub enum Guard {
    If(P<Expr>),
}

#[derive(RustcEncodable, RustcDecodable, Debug, HashStable)]
pub struct Field {
    #[stable_hasher(ignore)]
    pub hir_id: HirId,
    pub ident: Ident,
    pub expr: P<Expr>,
    pub span: Span,
    pub is_shorthand: bool,
}

#[derive(Copy, Clone, PartialEq, RustcEncodable, RustcDecodable, Debug, HashStable)]
pub enum BlockCheckMode {
    DefaultBlock,
    UnsafeBlock(UnsafeSource),
    PushUnsafeBlock(UnsafeSource),
    PopUnsafeBlock(UnsafeSource),
}

#[derive(Copy, Clone, PartialEq, RustcEncodable, RustcDecodable, Debug, HashStable)]
pub enum UnsafeSource {
    CompilerGenerated,
    UserProvided,
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, RustcEncodable, RustcDecodable, Hash, Debug)]
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
/// - an `arguments` array containing the `(x, y)` pattern
/// - a `value` containing the `x + y` expression (maybe wrapped in a block)
/// - `generator_kind` would be `None`
///
/// All bodies have an **owner**, which can be accessed via the HIR
/// map using `body_owner_def_id()`.
#[derive(RustcEncodable, RustcDecodable, Debug)]
pub struct Body {
    pub arguments: HirVec<Arg>,
    pub value: Expr,
    pub generator_kind: Option<GeneratorKind>,
}

impl Body {
    pub fn id(&self) -> BodyId {
        BodyId {
            hir_id: self.value.hir_id,
        }
    }
}

/// The type of source expression that caused this generator to be created.
// Not `IsAsync` because we want to eventually add support for `AsyncGen`
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, HashStable,
         RustcEncodable, RustcDecodable, Hash, Debug, Copy)]
pub enum GeneratorKind {
    /// An `async` block or function.
    Async,
    /// A generator literal created via a `yield` inside a closure.
    Gen,
}

impl fmt::Display for GeneratorKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            GeneratorKind::Async => "`async` object",
            GeneratorKind::Gen => "generator",
        })
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

/// A literal.
pub type Lit = Spanned<LitKind>;

/// A constant (expression) that's not an item or associated item,
/// but needs its own `DefId` for type-checking, const-eval, etc.
/// These are usually found nested inside types (e.g., array lengths)
/// or expressions (e.g., repeat counts), and also used to define
/// explicit discriminant values for enum variants.
#[derive(Copy, Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Debug, HashStable)]
pub struct AnonConst {
    pub hir_id: HirId,
    pub body: BodyId,
}

/// An expression
#[derive(RustcEncodable, RustcDecodable)]
pub struct Expr {
    pub span: Span,
    pub node: ExprKind,
    pub attrs: ThinVec<Attribute>,
    pub hir_id: HirId,
}

// `Expr` is used a lot. Make sure it doesn't unintentionally get bigger.
#[cfg(target_arch = "x86_64")]
static_assert_size!(Expr, 72);

impl Expr {
    pub fn precedence(&self) -> ExprPrecedence {
        match self.node {
            ExprKind::Box(_) => ExprPrecedence::Box,
            ExprKind::Array(_) => ExprPrecedence::Array,
            ExprKind::Call(..) => ExprPrecedence::Call,
            ExprKind::MethodCall(..) => ExprPrecedence::MethodCall,
            ExprKind::Tup(_) => ExprPrecedence::Tup,
            ExprKind::Binary(op, ..) => ExprPrecedence::Binary(op.node.into()),
            ExprKind::Unary(..) => ExprPrecedence::Unary,
            ExprKind::Lit(_) => ExprPrecedence::Lit,
            ExprKind::Type(..) | ExprKind::Cast(..) => ExprPrecedence::Cast,
            ExprKind::DropTemps(ref expr, ..) => expr.precedence(),
            ExprKind::While(..) => ExprPrecedence::While,
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
            ExprKind::Struct(..) => ExprPrecedence::Struct,
            ExprKind::Repeat(..) => ExprPrecedence::Repeat,
            ExprKind::Yield(..) => ExprPrecedence::Yield,
            ExprKind::Err => ExprPrecedence::Err,
        }
    }

    pub fn is_place_expr(&self) -> bool {
         match self.node {
            ExprKind::Path(QPath::Resolved(_, ref path)) => {
                match path.res {
                    Res::Local(..)
                    | Res::Def(DefKind::Static, _)
                    | Res::Err => true,
                    _ => false,
                }
            }

            ExprKind::Type(ref e, _) => {
                e.is_place_expr()
            }

            ExprKind::Unary(UnDeref, _) |
            ExprKind::Field(..) |
            ExprKind::Index(..) => {
                true
            }

            // Partially qualified paths in expressions can only legally
            // refer to associated items which are always rvalues.
            ExprKind::Path(QPath::TypeRelative(..)) |

            ExprKind::Call(..) |
            ExprKind::MethodCall(..) |
            ExprKind::Struct(..) |
            ExprKind::Tup(..) |
            ExprKind::Match(..) |
            ExprKind::Closure(..) |
            ExprKind::Block(..) |
            ExprKind::Repeat(..) |
            ExprKind::Array(..) |
            ExprKind::Break(..) |
            ExprKind::Continue(..) |
            ExprKind::Ret(..) |
            ExprKind::While(..) |
            ExprKind::Loop(..) |
            ExprKind::Assign(..) |
            ExprKind::InlineAsm(..) |
            ExprKind::AssignOp(..) |
            ExprKind::Lit(_) |
            ExprKind::Unary(..) |
            ExprKind::Box(..) |
            ExprKind::AddrOf(..) |
            ExprKind::Binary(..) |
            ExprKind::Yield(..) |
            ExprKind::Cast(..) |
            ExprKind::DropTemps(..) |
            ExprKind::Err => {
                false
            }
        }
    }
}

impl fmt::Debug for Expr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "expr({}: {})", self.hir_id,
               print::to_string(print::NO_ANN, |s| s.print_expr(self)))
    }
}

#[derive(RustcEncodable, RustcDecodable, Debug, HashStable)]
pub enum ExprKind {
    /// A `box x` expression.
    Box(P<Expr>),
    /// An array (e.g., `[a, b, c, d]`).
    Array(HirVec<Expr>),
    /// A function call.
    ///
    /// The first field resolves to the function itself (usually an `ExprKind::Path`),
    /// and the second field is the list of arguments.
    /// This also represents calling the constructor of
    /// tuple-like ADTs such as tuple structs and enum variants.
    Call(P<Expr>, HirVec<Expr>),
    /// A method call (e.g., `x.foo::<'static, Bar, Baz>(a, b, c, d)`).
    ///
    /// The `PathSegment`/`Span` represent the method name and its generic arguments
    /// (within the angle brackets).
    /// The first element of the vector of `Expr`s is the expression that evaluates
    /// to the object on which the method is being called on (the receiver),
    /// and the remaining elements are the rest of the arguments.
    /// Thus, `x.foo::<Bar, Baz>(a, b, c, d)` is represented as
    /// `ExprKind::MethodCall(PathSegment { foo, [Bar, Baz] }, [x, a, b, c, d])`.
    MethodCall(P<PathSegment>, Span, HirVec<Expr>),
    /// A tuple (e.g., `(a, b, c ,d)`).
    Tup(HirVec<Expr>),
    /// A binary operation (e.g., `a + b`, `a * b`).
    Binary(BinOp, P<Expr>, P<Expr>),
    /// A unary operation (e.g., `!x`, `*x`).
    Unary(UnOp, P<Expr>),
    /// A literal (e.g., `1`, `"foo"`).
    Lit(Lit),
    /// A cast (e.g., `foo as f64`).
    Cast(P<Expr>, P<Ty>),
    /// A type reference (e.g., `Foo`).
    Type(P<Expr>, P<Ty>),
    /// Wraps the expression in a terminating scope.
    /// This makes it semantically equivalent to `{ let _t = expr; _t }`.
    ///
    /// This construct only exists to tweak the drop order in HIR lowering.
    /// An example of that is the desugaring of `for` loops.
    DropTemps(P<Expr>),
    /// A while loop, with an optional label
    ///
    /// I.e., `'label: while expr { <block> }`.
    While(P<Expr>, P<Block>, Option<Label>),
    /// A conditionless loop (can be exited with `break`, `continue`, or `return`).
    ///
    /// I.e., `'label: loop { <block> }`.
    Loop(P<Block>, Option<Label>, LoopSource),
    /// A `match` block, with a source that indicates whether or not it is
    /// the result of a desugaring, and if so, which kind.
    Match(P<Expr>, HirVec<Arm>, MatchSource),
    /// A closure (e.g., `move |a, b, c| {a + b + c}`).
    ///
    /// The final span is the span of the argument block `|...|`.
    ///
    /// This may also be a generator literal or an `async block` as indicated by the
    /// `Option<GeneratorMovability>`.
    Closure(CaptureClause, P<FnDecl>, BodyId, Span, Option<GeneratorMovability>),
    /// A block (e.g., `'label: { ... }`).
    Block(P<Block>, Option<Label>),

    /// An assignment (e.g., `a = foo()`).
    Assign(P<Expr>, P<Expr>),
    /// An assignment with an operator.
    ///
    /// E.g., `a += 1`.
    AssignOp(BinOp, P<Expr>, P<Expr>),
    /// Access of a named (e.g., `obj.foo`) or unnamed (e.g., `obj.0`) struct or tuple field.
    Field(P<Expr>, Ident),
    /// An indexing operation (`foo[2]`).
    Index(P<Expr>, P<Expr>),

    /// Path to a definition, possibly containing lifetime or type parameters.
    Path(QPath),

    /// A referencing operation (i.e., `&a` or `&mut a`).
    AddrOf(Mutability, P<Expr>),
    /// A `break`, with an optional label to break.
    Break(Destination, Option<P<Expr>>),
    /// A `continue`, with an optional label.
    Continue(Destination),
    /// A `return`, with an optional value to be returned.
    Ret(Option<P<Expr>>),

    /// Inline assembly (from `asm!`), with its outputs and inputs.
    InlineAsm(P<InlineAsm>, HirVec<Expr>, HirVec<Expr>),

    /// A struct or struct-like variant literal expression.
    ///
    /// E.g., `Foo {x: 1, y: 2}`, or `Foo {x: 1, .. base}`,
    /// where `base` is the `Option<Expr>`.
    Struct(P<QPath>, HirVec<Field>, Option<P<Expr>>),

    /// An array literal constructed from one repeated element.
    ///
    /// E.g., `[1; 5]`. The first expression is the element
    /// to be repeated; the second is the number of times to repeat it.
    Repeat(P<Expr>, AnonConst),

    /// A suspension point for generators (i.e., `yield <expr>`).
    Yield(P<Expr>, YieldSource),

    /// A placeholder for an expression that wasn't syntactically well formed in some way.
    Err,
}

/// Represents an optionally `Self`-qualified value/type path or associated extension.
#[derive(RustcEncodable, RustcDecodable, Debug, HashStable)]
pub enum QPath {
    /// Path to a definition, optionally "fully-qualified" with a `Self`
    /// type, if the path points to an associated item in a trait.
    ///
    /// E.g., an unqualified path like `Clone::clone` has `None` for `Self`,
    /// while `<Vec<T> as Clone>::clone` has `Some(Vec<T>)` for `Self`,
    /// even though they both have the same two-segment `Clone::clone` `Path`.
    Resolved(Option<P<Ty>>, P<Path>),

    /// Type-related paths (e.g., `<T>::default` or `<T>::Output`).
    /// Will be resolved by type-checking to an associated item.
    ///
    /// UFCS source paths can desugar into this, with `Vec::new` turning into
    /// `<Vec>::new`, and `T::X::Y::method` into `<<<T>::X>::Y>::method`,
    /// the `X` and `Y` nodes each being a `TyKind::Path(QPath::TypeRelative(..))`.
    TypeRelative(P<Ty>, P<PathSegment>)
}

/// Hints at the original code for a let statement.
#[derive(Copy, Clone, RustcEncodable, RustcDecodable, Debug, HashStable)]
pub enum LocalSource {
    /// A `match _ { .. }`.
    Normal,
    /// A desugared `for _ in _ { .. }` loop.
    ForLoopDesugar,
    /// When lowering async functions, we create locals within the `async move` so that
    /// all arguments are dropped after the future is polled.
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
}

/// Hints at the original code for a `match _ { .. }`.
#[derive(Copy, Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Debug, HashStable)]
pub enum MatchSource {
    /// A `match _ { .. }`.
    Normal,
    /// An `if _ { .. }` (optionally with `else { .. }`).
    IfDesugar {
        contains_else_clause: bool,
    },
    /// An `if let _ = _ { .. }` (optionally with `else { .. }`).
    IfLetDesugar {
        contains_else_clause: bool,
    },
    /// A `while let _ = _ { .. }` (which was desugared to a
    /// `loop { match _ { .. } }`).
    WhileLetDesugar,
    /// A desugared `for _ in _ { .. }` loop.
    ForLoopDesugar,
    /// A desugared `?` operator.
    TryDesugar,
    /// A desugared `<expr>.await`.
    AwaitDesugar,
}

/// The loop type that yielded an `ExprKind::Loop`.
#[derive(Copy, Clone, PartialEq, RustcEncodable, RustcDecodable, Debug, HashStable)]
pub enum LoopSource {
    /// A `loop { .. }` loop.
    Loop,
    /// A `while let _ = _ { .. }` loop.
    WhileLet,
    /// A `for _ in _ { .. }` loop.
    ForLoop,
}

#[derive(Copy, Clone, RustcEncodable, RustcDecodable, Debug, HashStable)]
pub enum LoopIdError {
    OutsideLoopScope,
    UnlabeledCfInWhileCondition,
    UnresolvedLabel,
}

impl fmt::Display for LoopIdError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            LoopIdError::OutsideLoopScope => "not inside loop scope",
            LoopIdError::UnlabeledCfInWhileCondition =>
                "unlabeled control flow (break or continue) in while condition",
            LoopIdError::UnresolvedLabel => "label not found",
        })
    }
}

#[derive(Copy, Clone, RustcEncodable, RustcDecodable, Debug, HashStable)]
pub struct Destination {
    // This is `Some(_)` iff there is an explicit user-specified `label
    pub label: Option<Label>,

    // These errors are caught and then reported during the diagnostics pass in
    // librustc_passes/loops.rs
    pub target_id: Result<HirId, LoopIdError>,
}

/// Whether a generator contains self-references, causing it to be `!Unpin`.
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, HashStable,
         RustcEncodable, RustcDecodable, Hash, Debug)]
pub enum GeneratorMovability {
    /// May contain self-references, `!Unpin`.
    Static,
    /// Must not contain self-references, `Unpin`.
    Movable,
}

/// The yield kind that caused an `ExprKind::Yield`.
#[derive(Copy, Clone, PartialEq, Eq, Debug, RustcEncodable, RustcDecodable, HashStable)]
pub enum YieldSource {
    /// An `<expr>.await`.
    Await,
    /// A plain `yield`.
    Yield,
}

impl fmt::Display for YieldSource {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            YieldSource::Await => "`await`",
            YieldSource::Yield => "`yield`",
        })
    }
}

#[derive(Copy, Clone, RustcEncodable, RustcDecodable, Debug, HashStable)]
pub enum CaptureClause {
    CaptureByValue,
    CaptureByRef,
}

// N.B., if you change this, you'll probably want to change the corresponding
// type structure in middle/ty.rs as well.
#[derive(RustcEncodable, RustcDecodable, Debug, HashStable)]
pub struct MutTy {
    pub ty: P<Ty>,
    pub mutbl: Mutability,
}

/// Represents a method's signature in a trait declaration or implementation.
#[derive(RustcEncodable, RustcDecodable, Debug, HashStable)]
pub struct MethodSig {
    pub header: FnHeader,
    pub decl: P<FnDecl>,
}

// The bodies for items are stored "out of line", in a separate
// hashmap in the `Crate`. Here we just record the node-id of the item
// so it can fetched later.
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, RustcEncodable, RustcDecodable, Debug)]
pub struct TraitItemId {
    pub hir_id: HirId,
}

/// Represents an item declaration within a trait declaration,
/// possibly including a default implementation. A trait item is
/// either required (meaning it doesn't have an implementation, just a
/// signature) or provided (meaning it has a default implementation).
#[derive(RustcEncodable, RustcDecodable, Debug)]
pub struct TraitItem {
    pub ident: Ident,
    pub hir_id: HirId,
    pub attrs: HirVec<Attribute>,
    pub generics: Generics,
    pub node: TraitItemKind,
    pub span: Span,
}

/// Represents a trait method's body (or just argument names).
#[derive(RustcEncodable, RustcDecodable, Debug, HashStable)]
pub enum TraitMethod {
    /// No default body in the trait, just a signature.
    Required(HirVec<Ident>),

    /// Both signature and body are provided in the trait.
    Provided(BodyId),
}

/// Represents a trait method or associated constant or type
#[derive(RustcEncodable, RustcDecodable, Debug, HashStable)]
pub enum TraitItemKind {
    /// An associated constant with an optional value (otherwise `impl`s must contain a value).
    Const(P<Ty>, Option<BodyId>),
    /// A method with an optional body.
    Method(MethodSig, TraitMethod),
    /// An associated type with (possibly empty) bounds and optional concrete
    /// type.
    Type(GenericBounds, Option<P<Ty>>),
}

// The bodies for items are stored "out of line", in a separate
// hashmap in the `Crate`. Here we just record the node-id of the item
// so it can fetched later.
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, RustcEncodable, RustcDecodable, Debug)]
pub struct ImplItemId {
    pub hir_id: HirId,
}

/// Represents anything within an `impl` block
#[derive(RustcEncodable, RustcDecodable, Debug)]
pub struct ImplItem {
    pub ident: Ident,
    pub hir_id: HirId,
    pub vis: Visibility,
    pub defaultness: Defaultness,
    pub attrs: HirVec<Attribute>,
    pub generics: Generics,
    pub node: ImplItemKind,
    pub span: Span,
}

/// Represents various kinds of content within an `impl`.
#[derive(RustcEncodable, RustcDecodable, Debug, HashStable)]
pub enum ImplItemKind {
    /// An associated constant of the given type, set to the constant result
    /// of the expression
    Const(P<Ty>, BodyId),
    /// A method implementation with the given signature and body
    Method(MethodSig, BodyId),
    /// An associated type
    Type(P<Ty>),
    /// An associated existential type
    Existential(GenericBounds),
}

/// Bind a type to an associated type (i.e., `A = Foo`).
///
/// Bindings like `A: Debug` are represented as a special type `A =
/// $::Debug` that is understood by the astconv code.
///
/// FIXME(alexreg) -- why have a separate type for the binding case,
/// wouldn't it be better to make the `ty` field an enum like:
///
/// ```
/// enum TypeBindingKind {
///    Equals(...),
///    Binding(...),
/// }
/// ```
#[derive(RustcEncodable, RustcDecodable, Debug, HashStable)]
pub struct TypeBinding {
    pub hir_id: HirId,
    #[stable_hasher(project(name))]
    pub ident: Ident,
    pub kind: TypeBindingKind,
    pub span: Span,
}

// Represents the two kinds of type bindings.
#[derive(RustcEncodable, RustcDecodable, Debug, HashStable)]
pub enum TypeBindingKind {
    /// E.g., `Foo<Bar: Send>`.
    Constraint {
        bounds: HirVec<GenericBound>,
    },
    /// E.g., `Foo<Bar = ()>`.
    Equality {
        ty: P<Ty>,
    },
}

impl TypeBinding {
    pub fn ty(&self) -> &Ty {
        match self.kind {
            TypeBindingKind::Equality { ref ty } => ty,
            _ => bug!("expected equality type binding for parenthesized generic args"),
        }
    }
}

#[derive(RustcEncodable, RustcDecodable)]
pub struct Ty {
    pub hir_id: HirId,
    pub node: TyKind,
    pub span: Span,
}

impl fmt::Debug for Ty {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "type({})",
               print::to_string(print::NO_ANN, |s| s.print_type(self)))
    }
}

/// Not represented directly in the AST; referred to by name through a `ty_path`.
#[derive(Copy, Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Debug, HashStable)]
pub enum PrimTy {
    Int(IntTy),
    Uint(UintTy),
    Float(FloatTy),
    Str,
    Bool,
    Char,
}

#[derive(RustcEncodable, RustcDecodable, Debug, HashStable)]
pub struct BareFnTy {
    pub unsafety: Unsafety,
    pub abi: Abi,
    pub generic_params: HirVec<GenericParam>,
    pub decl: P<FnDecl>,
    pub arg_names: HirVec<Ident>,
}

#[derive(RustcEncodable, RustcDecodable, Debug, HashStable)]
pub struct ExistTy {
    pub generics: Generics,
    pub bounds: GenericBounds,
    pub impl_trait_fn: Option<DefId>,
    pub origin: ExistTyOrigin,
}

/// Where the existential type came from
#[derive(Copy, Clone, RustcEncodable, RustcDecodable, Debug, HashStable)]
pub enum ExistTyOrigin {
    /// `existential type Foo: Trait;`
    ExistentialType,
    /// `-> impl Trait`
    ReturnImplTrait,
    /// `async fn`
    AsyncFn,
}

/// The various kinds of types recognized by the compiler.
#[derive(RustcEncodable, RustcDecodable, Debug, HashStable)]
pub enum TyKind {
    /// A variable length slice (i.e., `[T]`).
    Slice(P<Ty>),
    /// A fixed length array (i.e., `[T; n]`).
    Array(P<Ty>, AnonConst),
    /// A raw pointer (i.e., `*const T` or `*mut T`).
    Ptr(MutTy),
    /// A reference (i.e., `&'a T` or `&'a mut T`).
    Rptr(Lifetime, MutTy),
    /// A bare function (e.g., `fn(usize) -> bool`).
    BareFn(P<BareFnTy>),
    /// The never type (`!`).
    Never,
    /// A tuple (`(A, B, C, D, ...)`).
    Tup(HirVec<Ty>),
    /// A path to a type definition (`module::module::...::Type`), or an
    /// associated type (e.g., `<Vec<T> as Trait>::Type` or `<T>::Target`).
    ///
    /// Type parameters may be stored in each `PathSegment`.
    Path(QPath),
    /// A type definition itself. This is currently only used for the `existential type`
    /// item that `impl Trait` in return position desugars to.
    ///
    /// The generic argument list contains the lifetimes (and in the future possibly parameters)
    /// that are actually bound on the `impl Trait`.
    Def(ItemId, HirVec<GenericArg>),
    /// A trait object type `Bound1 + Bound2 + Bound3`
    /// where `Bound` is a trait or a lifetime.
    TraitObject(HirVec<PolyTraitRef>, Lifetime),
    /// Unused for now.
    Typeof(AnonConst),
    /// `TyKind::Infer` means the type should be inferred instead of it having been
    /// specified. This can appear anywhere in a type.
    Infer,
    /// Placeholder for a type that has failed to be defined.
    Err,
    /// Placeholder for C-variadic arguments. We "spoof" the `VaListImpl` created
    /// from the variadic arguments. This type is only valid up to typeck.
    CVarArgs(Lifetime),
}

#[derive(Copy, Clone, RustcEncodable, RustcDecodable, Debug, HashStable)]
pub struct InlineAsmOutput {
    pub constraint: Symbol,
    pub is_rw: bool,
    pub is_indirect: bool,
    pub span: Span,
}

// NOTE(eddyb) This is used within MIR as well, so unlike the rest of the HIR,
// it needs to be `Clone` and use plain `Vec<T>` instead of `HirVec<T>`.
#[derive(Clone, RustcEncodable, RustcDecodable, Debug, HashStable)]
pub struct InlineAsm {
    pub asm: Symbol,
    pub asm_str_style: StrStyle,
    pub outputs: Vec<InlineAsmOutput>,
    pub inputs: Vec<Symbol>,
    pub clobbers: Vec<Symbol>,
    pub volatile: bool,
    pub alignstack: bool,
    pub dialect: AsmDialect,
    #[stable_hasher(ignore)] // This is used for error reporting
    pub ctxt: SyntaxContext,
}

/// Represents an argument in a function header.
#[derive(RustcEncodable, RustcDecodable, Debug, HashStable)]
pub struct Arg {
    pub pat: P<Pat>,
    pub hir_id: HirId,
}

/// Represents the header (not the body) of a function declaration.
#[derive(RustcEncodable, RustcDecodable, Debug, HashStable)]
pub struct FnDecl {
    /// The types of the function's arguments.
    ///
    /// Additional argument data is stored in the function's [body](Body::arguments).
    pub inputs: HirVec<Ty>,
    pub output: FunctionRetTy,
    pub c_variadic: bool,
    /// Does the function have an implicit self?
    pub implicit_self: ImplicitSelfKind,
}

/// Represents what type of implicit self a function has, if any.
#[derive(Copy, Clone, RustcEncodable, RustcDecodable, Debug, HashStable)]
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
    None
}

impl ImplicitSelfKind {
    /// Does this represent an implicit self?
    pub fn has_implicit_self(&self) -> bool {
        match *self {
            ImplicitSelfKind::None => false,
            _ => true,
        }
    }
}

/// Is the trait definition an auto trait?
#[derive(Copy, Clone, PartialEq, RustcEncodable, RustcDecodable, Debug, HashStable)]
pub enum IsAuto {
    Yes,
    No
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, HashStable,
         Ord, RustcEncodable, RustcDecodable, Debug)]
pub enum IsAsync {
    Async,
    NotAsync,
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, HashStable,
         RustcEncodable, RustcDecodable, Hash, Debug)]
pub enum Unsafety {
    Unsafe,
    Normal,
}

#[derive(Copy, Clone, PartialEq, RustcEncodable, RustcDecodable, Debug, HashStable)]
pub enum Constness {
    Const,
    NotConst,
}

#[derive(Copy, Clone, PartialEq, RustcEncodable, RustcDecodable, Debug, HashStable)]
pub enum Defaultness {
    Default { has_value: bool },
    Final,
}

impl Defaultness {
    pub fn has_value(&self) -> bool {
        match *self {
            Defaultness::Default { has_value, .. } => has_value,
            Defaultness::Final => true,
        }
    }

    pub fn is_final(&self) -> bool {
        *self == Defaultness::Final
    }

    pub fn is_default(&self) -> bool {
        match *self {
            Defaultness::Default { .. } => true,
            _ => false,
        }
    }
}

impl fmt::Display for Unsafety {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Unsafety::Normal => "normal",
            Unsafety::Unsafe => "unsafe",
        })
    }
}

#[derive(Copy, Clone, PartialEq, RustcEncodable, RustcDecodable, HashStable)]
pub enum ImplPolarity {
    /// `impl Trait for Type`
    Positive,
    /// `impl !Trait for Type`
    Negative,
}

impl fmt::Debug for ImplPolarity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            ImplPolarity::Positive => "positive",
            ImplPolarity::Negative => "negative",
        })
    }
}


#[derive(RustcEncodable, RustcDecodable, Debug, HashStable)]
pub enum FunctionRetTy {
    /// Return type is not specified.
    ///
    /// Functions default to `()` and
    /// closures default to inference. Span points to where return
    /// type would be inserted.
    DefaultReturn(Span),
    /// Everything else.
    Return(P<Ty>),
}

impl fmt::Display for FunctionRetTy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Return(ref ty) => print::to_string(print::NO_ANN, |s| s.print_type(ty)).fmt(f),
            DefaultReturn(_) => "()".fmt(f),
        }
    }
}

impl FunctionRetTy {
    pub fn span(&self) -> Span {
        match *self {
            DefaultReturn(span) => span,
            Return(ref ty) => ty.span,
        }
    }
}

#[derive(RustcEncodable, RustcDecodable, Debug)]
pub struct Mod {
    /// A span from the first token past `{` to the last token until `}`.
    /// For `mod foo;`, the inner span ranges from the first token
    /// to the last token in the external file.
    pub inner: Span,
    pub item_ids: HirVec<ItemId>,
}

#[derive(RustcEncodable, RustcDecodable, Debug, HashStable)]
pub struct ForeignMod {
    pub abi: Abi,
    pub items: HirVec<ForeignItem>,
}

#[derive(RustcEncodable, RustcDecodable, Debug, HashStable)]
pub struct GlobalAsm {
    pub asm: Symbol,
    #[stable_hasher(ignore)] // This is used for error reporting
    pub ctxt: SyntaxContext,
}

#[derive(RustcEncodable, RustcDecodable, Debug, HashStable)]
pub struct EnumDef {
    pub variants: HirVec<Variant>,
}

#[derive(RustcEncodable, RustcDecodable, Debug, HashStable)]
pub struct VariantKind {
    /// Name of the variant.
    #[stable_hasher(project(name))]
    pub ident: Ident,
    /// Attributes of the variant.
    pub attrs: HirVec<Attribute>,
    /// Id of the variant (not the constructor, see `VariantData::ctor_hir_id()`).
    pub id: HirId,
    /// Fields and constructor id of the variant.
    pub data: VariantData,
    /// Explicit discriminant (e.g., `Foo = 1`).
    pub disr_expr: Option<AnonConst>,
}

pub type Variant = Spanned<VariantKind>;

#[derive(Copy, Clone, PartialEq, RustcEncodable, RustcDecodable, Debug, HashStable)]
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
#[derive(RustcEncodable, RustcDecodable, Debug, HashStable)]
pub struct TraitRef {
    pub path: P<Path>,
    // Don't hash the ref_id. It is tracked via the thing it is used to access
    #[stable_hasher(ignore)]
    pub hir_ref_id: HirId,
}

impl TraitRef {
    /// Gets the `DefId` of the referenced trait. It _must_ actually be a trait or trait alias.
    pub fn trait_def_id(&self) -> DefId {
        match self.path.res {
            Res::Def(DefKind::Trait, did) => did,
            Res::Def(DefKind::TraitAlias, did) => did,
            Res::Err => {
                FatalError.raise();
            }
            _ => unreachable!(),
        }
    }
}

#[derive(RustcEncodable, RustcDecodable, Debug, HashStable)]
pub struct PolyTraitRef {
    /// The `'a` in `<'a> Foo<&'a T>`.
    pub bound_generic_params: HirVec<GenericParam>,

    /// The `Foo<&'a T>` in `<'a> Foo<&'a T>`.
    pub trait_ref: TraitRef,

    pub span: Span,
}

pub type Visibility = Spanned<VisibilityKind>;

#[derive(RustcEncodable, RustcDecodable, Debug)]
pub enum VisibilityKind {
    Public,
    Crate(CrateSugar),
    Restricted { path: P<Path>, hir_id: HirId },
    Inherited,
}

impl VisibilityKind {
    pub fn is_pub(&self) -> bool {
        match *self {
            VisibilityKind::Public => true,
            _ => false
        }
    }

    pub fn is_pub_restricted(&self) -> bool {
        match *self {
            VisibilityKind::Public |
            VisibilityKind::Inherited => false,
            VisibilityKind::Crate(..) |
            VisibilityKind::Restricted { .. } => true,
        }
    }

    pub fn descr(&self) -> &'static str {
        match *self {
            VisibilityKind::Public => "public",
            VisibilityKind::Inherited => "private",
            VisibilityKind::Crate(..) => "crate-visible",
            VisibilityKind::Restricted { .. } => "restricted",
        }
    }
}

#[derive(RustcEncodable, RustcDecodable, Debug, HashStable)]
pub struct StructField {
    pub span: Span,
    #[stable_hasher(project(name))]
    pub ident: Ident,
    pub vis: Visibility,
    pub hir_id: HirId,
    pub ty: P<Ty>,
    pub attrs: HirVec<Attribute>,
}

impl StructField {
    // Still necessary in couple of places
    pub fn is_positional(&self) -> bool {
        let first = self.ident.as_str().as_bytes()[0];
        first >= b'0' && first <= b'9'
    }
}

/// Fields and constructor IDs of enum variants and structs.
#[derive(RustcEncodable, RustcDecodable, Debug, HashStable)]
pub enum VariantData {
    /// A struct variant.
    ///
    /// E.g., `Bar { .. }` as in `enum Foo { Bar { .. } }`.
    Struct(HirVec<StructField>, /* recovered */ bool),
    /// A tuple variant.
    ///
    /// E.g., `Bar(..)` as in `enum Foo { Bar(..) }`.
    Tuple(HirVec<StructField>, HirId),
    /// A unit variant.
    ///
    /// E.g., `Bar = ..` as in `enum Foo { Bar = .. }`.
    Unit(HirId),
}

impl VariantData {
    /// Return the fields of this variant.
    pub fn fields(&self) -> &[StructField] {
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
// hashmap in the `Crate`. Here we just record the node-id of the item
// so it can fetched later.
#[derive(Copy, Clone, RustcEncodable, RustcDecodable, Debug)]
pub struct ItemId {
    pub id: HirId,
}

/// An item
///
/// The name might be a dummy name in case of anonymous items
#[derive(RustcEncodable, RustcDecodable, Debug)]
pub struct Item {
    pub ident: Ident,
    pub hir_id: HirId,
    pub attrs: HirVec<Attribute>,
    pub node: ItemKind,
    pub vis: Visibility,
    pub span: Span,
}

#[derive(Copy, Clone, RustcEncodable, RustcDecodable, Debug, HashStable)]
pub struct FnHeader {
    pub unsafety: Unsafety,
    pub constness: Constness,
    pub asyncness: IsAsync,
    pub abi: Abi,
}

impl FnHeader {
    pub fn is_const(&self) -> bool {
        match &self.constness {
            Constness::Const => true,
            _ => false,
        }
    }
}

#[derive(RustcEncodable, RustcDecodable, Debug, HashStable)]
pub enum ItemKind {
    /// An `extern crate` item, with optional *original* crate name if the crate was renamed.
    ///
    /// E.g., `extern crate foo` or `extern crate foo_bar as foo`.
    ExternCrate(Option<Name>),

    /// `use foo::bar::*;` or `use foo::bar::baz as quux;`
    ///
    /// or just
    ///
    /// `use foo::bar::baz;` (with `as baz` implicitly on the right)
    Use(P<Path>, UseKind),

    /// A `static` item
    Static(P<Ty>, Mutability, BodyId),
    /// A `const` item
    Const(P<Ty>, BodyId),
    /// A function declaration
    Fn(P<FnDecl>, FnHeader, Generics, BodyId),
    /// A module
    Mod(Mod),
    /// An external module
    ForeignMod(ForeignMod),
    /// Module-level inline assembly (from global_asm!)
    GlobalAsm(P<GlobalAsm>),
    /// A type alias, e.g., `type Foo = Bar<u8>`
    Ty(P<Ty>, Generics),
    /// An existential type definition, e.g., `existential type Foo: Bar;`
    Existential(ExistTy),
    /// An enum definition, e.g., `enum Foo<A, B> {C<A>, D<B>}`
    Enum(EnumDef, Generics),
    /// A struct definition, e.g., `struct Foo<A> {x: A}`
    Struct(VariantData, Generics),
    /// A union definition, e.g., `union Foo<A, B> {x: A, y: B}`
    Union(VariantData, Generics),
    /// Represents a Trait Declaration
    Trait(IsAuto, Unsafety, Generics, GenericBounds, HirVec<TraitItemRef>),
    /// Represents a Trait Alias Declaration
    TraitAlias(Generics, GenericBounds),

    /// An implementation, eg `impl<A> Trait for Foo { .. }`
    Impl(Unsafety,
         ImplPolarity,
         Defaultness,
         Generics,
         Option<TraitRef>, // (optional) trait this impl implements
         P<Ty>, // self
         HirVec<ImplItemRef>),
}

impl ItemKind {
    pub fn descriptive_variant(&self) -> &str {
        match *self {
            ItemKind::ExternCrate(..) => "extern crate",
            ItemKind::Use(..) => "use",
            ItemKind::Static(..) => "static item",
            ItemKind::Const(..) => "constant item",
            ItemKind::Fn(..) => "function",
            ItemKind::Mod(..) => "module",
            ItemKind::ForeignMod(..) => "foreign module",
            ItemKind::GlobalAsm(..) => "global asm",
            ItemKind::Ty(..) => "type alias",
            ItemKind::Existential(..) => "existential type",
            ItemKind::Enum(..) => "enum",
            ItemKind::Struct(..) => "struct",
            ItemKind::Union(..) => "union",
            ItemKind::Trait(..) => "trait",
            ItemKind::TraitAlias(..) => "trait alias",
            ItemKind::Impl(..) => "impl",
        }
    }

    pub fn adt_kind(&self) -> Option<AdtKind> {
        match *self {
            ItemKind::Struct(..) => Some(AdtKind::Struct),
            ItemKind::Union(..) => Some(AdtKind::Union),
            ItemKind::Enum(..) => Some(AdtKind::Enum),
            _ => None,
        }
    }

    pub fn generics(&self) -> Option<&Generics> {
        Some(match *self {
            ItemKind::Fn(_, _, ref generics, _) |
            ItemKind::Ty(_, ref generics) |
            ItemKind::Existential(ExistTy { ref generics, impl_trait_fn: None, .. }) |
            ItemKind::Enum(_, ref generics) |
            ItemKind::Struct(_, ref generics) |
            ItemKind::Union(_, ref generics) |
            ItemKind::Trait(_, _, ref generics, _, _) |
            ItemKind::Impl(_, _, _, ref generics, _, _, _)=> generics,
            _ => return None
        })
    }
}

/// A reference from an trait to one of its associated items. This
/// contains the item's id, naturally, but also the item's name and
/// some other high-level details (like whether it is an associated
/// type or method, and whether it is public). This allows other
/// passes to find the impl they want without loading the ID (which
/// means fewer edges in the incremental compilation graph).
#[derive(RustcEncodable, RustcDecodable, Debug, HashStable)]
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
#[derive(RustcEncodable, RustcDecodable, Debug, HashStable)]
pub struct ImplItemRef {
    pub id: ImplItemId,
    #[stable_hasher(project(name))]
    pub ident: Ident,
    pub kind: AssocItemKind,
    pub span: Span,
    pub vis: Visibility,
    pub defaultness: Defaultness,
}

#[derive(Copy, Clone, PartialEq, RustcEncodable, RustcDecodable, Debug, HashStable)]
pub enum AssocItemKind {
    Const,
    Method { has_self: bool },
    Type,
    Existential,
}

#[derive(RustcEncodable, RustcDecodable, Debug, HashStable)]
pub struct ForeignItem {
    #[stable_hasher(project(name))]
    pub ident: Ident,
    pub attrs: HirVec<Attribute>,
    pub node: ForeignItemKind,
    pub hir_id: HirId,
    pub span: Span,
    pub vis: Visibility,
}

/// An item within an `extern` block.
#[derive(RustcEncodable, RustcDecodable, Debug, HashStable)]
pub enum ForeignItemKind {
    /// A foreign function.
    Fn(P<FnDecl>, HirVec<Ident>, Generics),
    /// A foreign static item (`static ext: u8`).
    Static(P<Ty>, Mutability),
    /// A foreign type.
    Type,
}

impl ForeignItemKind {
    pub fn descriptive_variant(&self) -> &str {
        match *self {
            ForeignItemKind::Fn(..) => "foreign function",
            ForeignItemKind::Static(..) => "foreign static item",
            ForeignItemKind::Type => "foreign type",
        }
    }
}

/// A variable captured by a closure.
#[derive(Debug, Copy, Clone, RustcEncodable, RustcDecodable, HashStable)]
pub struct Upvar {
    // First span where it is accessed (there can be multiple).
    pub span: Span
}

pub type CaptureModeMap = NodeMap<CaptureClause>;

 // The TraitCandidate's import_ids is empty if the trait is defined in the same module, and
 // has length > 0 if the trait is found through an chain of imports, starting with the
 // import/use statement in the scope where the trait is used.
#[derive(Clone, Debug)]
pub struct TraitCandidate {
    pub def_id: DefId,
    pub import_ids: SmallVec<[NodeId; 1]>,
}

// Trait method resolution
pub type TraitMap = NodeMap<Vec<TraitCandidate>>;

// Map from the NodeId of a glob import to a list of items which are actually
// imported.
pub type GlobMap = NodeMap<FxHashSet<Name>>;

pub fn provide(providers: &mut Providers<'_>) {
    check_attr::provide(providers);
    map::provide(providers);
    upvars::provide(providers);
}

#[derive(Clone, RustcEncodable, RustcDecodable, HashStable)]
pub struct CodegenFnAttrs {
    pub flags: CodegenFnAttrFlags,
    /// Parsed representation of the `#[inline]` attribute
    pub inline: InlineAttr,
    /// Parsed representation of the `#[optimize]` attribute
    pub optimize: OptimizeAttr,
    /// The `#[export_name = "..."]` attribute, indicating a custom symbol a
    /// function should be exported under
    pub export_name: Option<Symbol>,
    /// The `#[link_name = "..."]` attribute, indicating a custom symbol an
    /// imported function should be imported as. Note that `export_name`
    /// probably isn't set when this is set, this is for foreign items while
    /// `#[export_name]` is for Rust-defined functions.
    pub link_name: Option<Symbol>,
    /// The `#[target_feature(enable = "...")]` attribute and the enabled
    /// features (only enabled features are supported right now).
    pub target_features: Vec<Symbol>,
    /// The `#[linkage = "..."]` attribute and the value we found.
    pub linkage: Option<Linkage>,
    /// The `#[link_section = "..."]` attribute, or what executable section this
    /// should be placed in.
    pub link_section: Option<Symbol>,
}

bitflags! {
    #[derive(RustcEncodable, RustcDecodable, HashStable)]
    pub struct CodegenFnAttrFlags: u32 {
        /// `#[cold]`: a hint to LLVM that this function, when called, is never on
        /// the hot path.
        const COLD                      = 1 << 0;
        /// `#[rustc_allocator]`: a hint to LLVM that the pointer returned from this
        /// function is never null.
        const ALLOCATOR                 = 1 << 1;
        /// `#[unwind]`: an indicator that this function may unwind despite what
        /// its ABI signature may otherwise imply.
        const UNWIND                    = 1 << 2;
        /// `#[rust_allocator_nounwind]`, an indicator that an imported FFI
        /// function will never unwind. Probably obsolete by recent changes with
        /// #[unwind], but hasn't been removed/migrated yet
        const RUSTC_ALLOCATOR_NOUNWIND  = 1 << 3;
        /// `#[naked]`: an indicator to LLVM that no function prologue/epilogue
        /// should be generated.
        const NAKED                     = 1 << 4;
        /// `#[no_mangle]`: an indicator that the function's name should be the same
        /// as its symbol.
        const NO_MANGLE                 = 1 << 5;
        /// `#[rustc_std_internal_symbol]`: an indicator that this symbol is a
        /// "weird symbol" for the standard library in that it has slightly
        /// different linkage, visibility, and reachability rules.
        const RUSTC_STD_INTERNAL_SYMBOL = 1 << 6;
        /// `#[no_debug]`: an indicator that no debugging information should be
        /// generated for this function by LLVM.
        const NO_DEBUG                  = 1 << 7;
        /// `#[thread_local]`: indicates a static is actually a thread local
        /// piece of memory
        const THREAD_LOCAL              = 1 << 8;
        /// `#[used]`: indicates that LLVM can't eliminate this function (but the
        /// linker can!).
        const USED                      = 1 << 9;
        /// #[ffi_returns_twice], indicates that an extern function can return
        /// multiple times
        const FFI_RETURNS_TWICE = 1 << 10;
    }
}

impl CodegenFnAttrs {
    pub fn new() -> CodegenFnAttrs {
        CodegenFnAttrs {
            flags: CodegenFnAttrFlags::empty(),
            inline: InlineAttr::None,
            optimize: OptimizeAttr::None,
            export_name: None,
            link_name: None,
            target_features: vec![],
            linkage: None,
            link_section: None,
        }
    }

    /// Returns `true` if `#[inline]` or `#[inline(always)]` is present.
    pub fn requests_inline(&self) -> bool {
        match self.inline {
            InlineAttr::Hint | InlineAttr::Always => true,
            InlineAttr::None | InlineAttr::Never => false,
        }
    }

    /// Returns `true` if it looks like this symbol needs to be exported, for example:
    ///
    /// * `#[no_mangle]` is present
    /// * `#[export_name(...)]` is present
    /// * `#[linkage]` is present
    pub fn contains_extern_indicator(&self) -> bool {
        self.flags.contains(CodegenFnAttrFlags::NO_MANGLE) ||
            self.export_name.is_some() ||
            match self.linkage {
                // These are private, so make sure we don't try to consider
                // them external.
                None |
                Some(Linkage::Internal) |
                Some(Linkage::Private) => false,
                Some(_) => true,
            }
    }
}

#[derive(Copy, Clone, Debug)]
pub enum Node<'hir> {
    Item(&'hir Item),
    ForeignItem(&'hir ForeignItem),
    TraitItem(&'hir TraitItem),
    ImplItem(&'hir ImplItem),
    Variant(&'hir Variant),
    Field(&'hir StructField),
    AnonConst(&'hir AnonConst),
    Expr(&'hir Expr),
    Stmt(&'hir Stmt),
    PathSegment(&'hir PathSegment),
    Ty(&'hir Ty),
    TraitRef(&'hir TraitRef),
    Binding(&'hir Pat),
    Pat(&'hir Pat),
    Arm(&'hir Arm),
    Block(&'hir Block),
    Local(&'hir Local),
    MacroDef(&'hir MacroDef),

    /// `Ctor` refers to the constructor of an enum variant or struct. Only tuple or unit variants
    /// with synthesized constructors.
    Ctor(&'hir VariantData),

    Lifetime(&'hir Lifetime),
    GenericParam(&'hir GenericParam),
    Visibility(&'hir Visibility),

    Crate,
}
