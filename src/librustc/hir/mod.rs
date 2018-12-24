// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// HIR datatypes. See the [rustc guide] for more info.
//!
//! [rustc guide]: https://rust-lang.github.io/rustc-guide/hir.html

pub use self::BlockCheckMode::*;
pub use self::CaptureClause::*;
pub use self::FunctionRetTy::*;
pub use self::Mutability::*;
pub use self::PrimTy::*;
pub use self::UnOp::*;
pub use self::UnsafeSource::*;

use hir::def::Def;
use hir::def_id::{DefId, DefIndex, LocalDefId, CRATE_DEF_INDEX};
use util::nodemap::{NodeMap, FxHashSet};
use mir::mono::Linkage;

use syntax_pos::{Span, DUMMY_SP, symbol::InternedString};
use syntax::source_map::{self, Spanned};
use rustc_target::spec::abi::Abi;
use syntax::ast::{self, CrateSugar, Ident, Name, NodeId, DUMMY_NODE_ID, AsmDialect};
use syntax::ast::{Attribute, Lit, StrStyle, FloatTy, IntTy, UintTy};
use syntax::attr::InlineAttr;
use syntax::ext::hygiene::SyntaxContext;
use syntax::ptr::P;
use syntax::symbol::{Symbol, keywords};
use syntax::tokenstream::TokenStream;
use syntax::util::parser::ExprPrecedence;
use ty::AdtKind;
use ty::query::Providers;

use rustc_data_structures::sync::{ParallelIterator, par_iter, Send, Sync, scope};
use rustc_data_structures::thin_vec::ThinVec;

use serialize::{self, Encoder, Encodable, Decoder, Decodable};
use std::collections::BTreeMap;
use std::fmt;

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

/// A HirId uniquely identifies a node in the HIR of the current crate. It is
/// composed of the `owner`, which is the DefIndex of the directly enclosing
/// hir::Item, hir::TraitItem, or hir::ImplItem (i.e., the closest "item-like"),
/// and the `local_id` which is unique within the given owner.
///
/// This two-level structure makes for more stable values: One can move an item
/// around within the source code, or add or remove stuff before it, without
/// the local_id part of the HirId changing, which is a very useful property in
/// incremental compilation where we have to persist things through changes to
/// the code base.
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
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

// hack to ensure that we don't try to access the private parts of `ItemLocalId` in this module
mod item_local_id_inner {
    use rustc_data_structures::indexed_vec::Idx;
    /// An `ItemLocalId` uniquely identifies something within a given "item-like",
    /// that is within a hir::Item, hir::TraitItem, or hir::ImplItem. There is no
    /// guarantee that the numerical value of a given `ItemLocalId` corresponds to
    /// the node's position within the owning item in any way, but there is a
    /// guarantee that the `LocalItemId`s within an owner occupy a dense range of
    /// integers starting at zero, so a mapping that maps all or most nodes within
    /// an "item-like" to something else can be implement by a `Vec` instead of a
    /// tree or hash map.
    newtype_index! {
        pub struct ItemLocalId { .. }
    }
}

pub use self::item_local_id_inner::ItemLocalId;

/// The `HirId` corresponding to CRATE_NODE_ID and CRATE_DEF_INDEX
pub const CRATE_HIR_ID: HirId = HirId {
    owner: CRATE_DEF_INDEX,
    local_id: ItemLocalId::from_u32_const(0)
};

pub const DUMMY_HIR_ID: HirId = HirId {
    owner: CRATE_DEF_INDEX,
    local_id: DUMMY_ITEM_LOCAL_ID,
};

pub const DUMMY_ITEM_LOCAL_ID: ItemLocalId = ItemLocalId::MAX;

#[derive(Clone, RustcEncodable, RustcDecodable, Copy)]
pub struct Label {
    pub ident: Ident,
}

impl fmt::Debug for Label {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "label({:?})", self.ident)
    }
}

#[derive(Clone, RustcEncodable, RustcDecodable, Copy)]
pub struct Lifetime {
    pub id: NodeId,
    pub span: Span,

    /// Either "'a", referring to a named lifetime definition,
    /// or "" (aka keywords::Invalid), for elision placeholders.
    ///
    /// HIR lowering inserts these placeholders in type paths that
    /// refer to type definitions needing lifetime parameters,
    /// `&T` and `&mut T`, and trait objects without `... + 'a`.
    pub name: LifetimeName,
}

#[derive(Debug, Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Copy)]
pub enum ParamName {
    /// Some user-given name like `T` or `'x`.
    Plain(Ident),

    /// Synthetic name generated when user elided a lifetime in an impl header,
    /// e.g., the lifetimes in cases like these:
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
    /// when e.g., `'_` is used in the wrong place.
    Error,
}

impl ParamName {
    pub fn ident(&self) -> Ident {
        match *self {
            ParamName::Plain(ident) => ident,
            ParamName::Error | ParamName::Fresh(_) => keywords::UnderscoreLifetime.ident(),
        }
    }

    pub fn modern(&self) -> ParamName {
        match *self {
            ParamName::Plain(ident) => ParamName::Plain(ident.modern()),
            param_name => param_name,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Copy)]
pub enum LifetimeName {
    /// User-given names or fresh (synthetic) names.
    Param(ParamName),

    /// User typed nothing. e.g., the lifetime in `&u32`.
    Implicit,

    /// Indicates an error during lowering (usually `'_` in wrong place)
    /// that was already reported.
    Error,

    /// User typed `'_`.
    Underscore,

    /// User wrote `'static`
    Static,
}

impl LifetimeName {
    pub fn ident(&self) -> Ident {
        match *self {
            LifetimeName::Implicit => keywords::Invalid.ident(),
            LifetimeName::Error => keywords::Invalid.ident(),
            LifetimeName::Underscore => keywords::UnderscoreLifetime.ident(),
            LifetimeName::Static => keywords::StaticLifetime.ident(),
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
               self.id,
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

/// A "Path" is essentially Rust's notion of a name; for instance:
/// `std::cmp::PartialEq`. It's represented as a sequence of identifiers,
/// along with a bunch of supporting information.
#[derive(Clone, RustcEncodable, RustcDecodable)]
pub struct Path {
    pub span: Span,
    /// The definition that the path resolved to.
    pub def: Def,
    /// The segments in the path: the things separated by `::`.
    pub segments: HirVec<PathSegment>,
}

impl Path {
    pub fn is_global(&self) -> bool {
        !self.segments.is_empty() && self.segments[0].ident.name == keywords::PathRoot.name()
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
#[derive(Clone, RustcEncodable, RustcDecodable, Debug)]
pub struct PathSegment {
    /// The identifier portion of this path segment.
    pub ident: Ident,
    // `id` and `def` are optional. We currently only use these in save-analysis,
    // any path segments without these will not have save-analysis info and
    // therefore will not have 'jump to def' in IDEs, but otherwise will not be
    // affected. (In general, we don't bother to get the defs for synthesized
    // segments, only for segments which have come from the AST).
    pub id: Option<NodeId>,
    pub def: Option<Def>,

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
    pub infer_types: bool,
}

impl PathSegment {
    /// Convert an identifier to the corresponding segment.
    pub fn from_ident(ident: Ident) -> PathSegment {
        PathSegment {
            ident,
            id: None,
            def: None,
            infer_types: true,
            args: None,
        }
    }

    pub fn new(
        ident: Ident,
        id: Option<NodeId>,
        def: Option<Def>,
        args: GenericArgs,
        infer_types: bool,
    ) -> Self {
        PathSegment {
            ident,
            id,
            def,
            infer_types,
            args: if args.is_empty() {
                None
            } else {
                Some(P(args))
            }
        }
    }

    // FIXME: hack required because you can't create a static
    // `GenericArgs`, so you can't just return a `&GenericArgs`.
    pub fn with_generic_args<F, R>(&self, f: F) -> R
        where F: FnOnce(&GenericArgs) -> R
    {
        let dummy = GenericArgs::none();
        f(if let Some(ref args) = self.args {
            &args
        } else {
            &dummy
        })
    }
}

#[derive(Clone, RustcEncodable, RustcDecodable, Debug)]
pub enum GenericArg {
    Lifetime(Lifetime),
    Type(Ty),
}

impl GenericArg {
    pub fn span(&self) -> Span {
        match self {
            GenericArg::Lifetime(l) => l.span,
            GenericArg::Type(t) => t.span,
        }
    }

    pub fn id(&self) -> NodeId {
        match self {
            GenericArg::Lifetime(l) => l.id,
            GenericArg::Type(t) => t.id,
        }
    }
}

#[derive(Clone, RustcEncodable, RustcDecodable, Debug)]
pub struct GenericArgs {
    /// The generic arguments for this path segment.
    pub args: HirVec<GenericArg>,
    /// Bindings (equality constraints) on associated types, if present.
    /// E.g., `Foo<A=Bar>`.
    pub bindings: HirVec<TypeBinding>,
    /// Were arguments written in parenthesized form `Fn(T) -> U`?
    /// This is required mostly for pretty-printing and diagnostics,
    /// but also for changing lifetime elision rules to be "function-like".
    pub parenthesized: bool,
}

impl GenericArgs {
    pub fn none() -> Self {
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
            };
        }

        own_counts
    }
}

/// A modifier on a bound, currently this is only used for `?Sized`, where the
/// modifier is `Maybe`. Negative bounds should also be handled here.
#[derive(Copy, Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Debug)]
pub enum TraitBoundModifier {
    None,
    Maybe,
}

/// The AST represents all type param bounds as types.
/// `typeck::collect::compute_bounds` matches these against
/// the "special" built-in traits (see `middle::lang_items`) and
/// detects `Copy`, `Send` and `Sync`.
#[derive(Clone, RustcEncodable, RustcDecodable, Debug)]
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

#[derive(Copy, Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Debug)]
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

#[derive(Clone, RustcEncodable, RustcDecodable, Debug)]
pub enum GenericParamKind {
    /// A lifetime definition (e.g., `'a: 'b + 'c + 'd`).
    Lifetime {
        kind: LifetimeParamKind,
    },
    Type {
        default: Option<P<Ty>>,
        synthetic: Option<SyntheticTyParamKind>,
    }
}

#[derive(Clone, RustcEncodable, RustcDecodable, Debug)]
pub struct GenericParam {
    pub id: NodeId,
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
}

/// Represents lifetimes and type parameters attached to a declaration
/// of a function, enum, trait, etc.
#[derive(Clone, RustcEncodable, RustcDecodable, Debug)]
pub struct Generics {
    pub params: HirVec<GenericParam>,
    pub where_clause: WhereClause,
    pub span: Span,
}

impl Generics {
    pub fn empty() -> Generics {
        Generics {
            params: HirVec::new(),
            where_clause: WhereClause {
                id: DUMMY_NODE_ID,
                predicates: HirVec::new(),
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
            };
        }

        own_counts
    }

    pub fn get_named(&self, name: &InternedString) -> Option<&GenericParam> {
        for param in &self.params {
            if *name == param.name.ident().as_interned_str() {
                return Some(param);
            }
        }
        None
    }
}

/// Synthetic Type Parameters are converted to an other form during lowering, this allows
/// to track the original form they had. Useful for error messages.
#[derive(Copy, Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Debug)]
pub enum SyntheticTyParamKind {
    ImplTrait
}

/// A `where` clause in a definition
#[derive(Clone, RustcEncodable, RustcDecodable, Debug)]
pub struct WhereClause {
    pub id: NodeId,
    pub predicates: HirVec<WherePredicate>,
}

impl WhereClause {
    pub fn span(&self) -> Option<Span> {
        self.predicates.iter().map(|predicate| predicate.span())
            .fold(None, |acc, i| match (acc, i) {
                (None, i) => Some(i),
                (Some(acc), i) => {
                    Some(acc.to(i))
                }
            })
    }
}

/// A single predicate in a `where` clause
#[derive(Clone, RustcEncodable, RustcDecodable, Debug)]
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

/// A type bound, eg `for<'c> Foo: Send+Clone+'c`
#[derive(Clone, RustcEncodable, RustcDecodable, Debug)]
pub struct WhereBoundPredicate {
    pub span: Span,
    /// Any generics from a `for` binding
    pub bound_generic_params: HirVec<GenericParam>,
    /// The type being bounded
    pub bounded_ty: P<Ty>,
    /// Trait and lifetime bounds (`Clone+Send+'static`)
    pub bounds: GenericBounds,
}

/// A lifetime predicate, e.g., `'a: 'b+'c`
#[derive(Clone, RustcEncodable, RustcDecodable, Debug)]
pub struct WhereRegionPredicate {
    pub span: Span,
    pub lifetime: Lifetime,
    pub bounds: GenericBounds,
}

/// An equality predicate (unsupported), e.g., `T=int`
#[derive(Clone, RustcEncodable, RustcDecodable, Debug)]
pub struct WhereEqPredicate {
    pub id: NodeId,
    pub span: Span,
    pub lhs_ty: P<Ty>,
    pub rhs_ty: P<Ty>,
}

/// The top-level data structure that stores the entire contents of
/// the crate currently being compiled.
///
/// For more details, see the [rustc guide].
///
/// [rustc guide]: https://rust-lang.github.io/rustc-guide/hir.html
#[derive(Clone, RustcEncodable, RustcDecodable, Debug)]
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
    pub items: BTreeMap<NodeId, Item>,

    pub trait_items: BTreeMap<TraitItemId, TraitItem>,
    pub impl_items: BTreeMap<ImplItemId, ImplItem>,
    pub bodies: BTreeMap<BodyId, Body>,
    pub trait_impls: BTreeMap<DefId, Vec<NodeId>>,
    pub trait_auto_impl: BTreeMap<DefId, NodeId>,

    /// A list of the body ids written out in the order in which they
    /// appear in the crate. If you're going to process all the bodies
    /// in the crate, you should iterate over this list rather than the keys
    /// of bodies.
    pub body_ids: Vec<BodyId>,
}

impl Crate {
    pub fn item(&self, id: NodeId) -> &Item {
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

    /// A parallel version of visit_all_item_likes
    pub fn par_visit_all_item_likes<'hir, V>(&'hir self, visitor: &V)
        where V: itemlikevisit::ParItemLikeVisitor<'hir> + Sync + Send
    {
        scope(|s| {
            s.spawn(|_| {
                par_iter(&self.items).for_each(|(_, item)| {
                    visitor.visit_item(item);
                });
            });

            s.spawn(|_| {
                par_iter(&self.trait_items).for_each(|(_, trait_item)| {
                    visitor.visit_trait_item(trait_item);
                });
            });

            s.spawn(|_| {
                par_iter(&self.impl_items).for_each(|(_, impl_item)| {
                    visitor.visit_impl_item(impl_item);
                });
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
#[derive(Clone, RustcEncodable, RustcDecodable, Debug)]
pub struct MacroDef {
    pub name: Name,
    pub vis: Visibility,
    pub attrs: HirVec<Attribute>,
    pub id: NodeId,
    pub span: Span,
    pub body: TokenStream,
    pub legacy: bool,
}

#[derive(Clone, RustcEncodable, RustcDecodable, Debug)]
pub struct Block {
    /// Statements in a block
    pub stmts: HirVec<Stmt>,
    /// An expression at the end of the block
    /// without a semicolon, if any
    pub expr: Option<P<Expr>>,
    pub id: NodeId,
    pub hir_id: HirId,
    /// Distinguishes between `unsafe { ... }` and `{ ... }`
    pub rules: BlockCheckMode,
    pub span: Span,
    /// If true, then there may exist `break 'a` values that aim to
    /// break out of this block early.
    /// Used by `'label: {}` blocks and by `catch` statements.
    pub targeted_by_break: bool,
    /// If true, don't emit return value type errors as the parser had
    /// to recover from a parse error so this block will not have an
    /// appropriate type. A parse error will have been emitted so the
    /// compilation will never succeed if this is true.
    pub recovered: bool,
}

#[derive(Clone, RustcEncodable, RustcDecodable)]
pub struct Pat {
    pub id: NodeId,
    pub hir_id: HirId,
    pub node: PatKind,
    pub span: Span,
}

impl fmt::Debug for Pat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "pat({}: {})", self.id,
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

/// A single field in a struct pattern
///
/// Patterns like the fields of Foo `{ x, ref y, ref mut z }`
/// are treated the same as` x: x, y: ref y, z: ref mut z`,
/// except is_shorthand is true
#[derive(Clone, RustcEncodable, RustcDecodable, Debug)]
pub struct FieldPat {
    pub id: NodeId,
    /// The identifier for the field
    pub ident: Ident,
    /// The pattern the field is destructured to
    pub pat: P<Pat>,
    pub is_shorthand: bool,
}

/// Explicit binding annotations given in the HIR for a binding. Note
/// that this is not the final binding *mode* that we infer after type
/// inference.
#[derive(Clone, PartialEq, RustcEncodable, RustcDecodable, Debug, Copy)]
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

#[derive(Copy, Clone, PartialEq, RustcEncodable, RustcDecodable, Debug)]
pub enum RangeEnd {
    Included,
    Excluded,
}

#[derive(Clone, RustcEncodable, RustcDecodable, Debug)]
pub enum PatKind {
    /// Represents a wildcard pattern (`_`)
    Wild,

    /// A fresh binding `ref mut binding @ OPT_SUBPATTERN`.
    /// The `NodeId` is the canonical ID for the variable being bound,
    /// e.g., in `Ok(x) | Err(x)`, both `x` use the same canonical ID,
    /// which is the pattern ID of the first `x`.
    Binding(BindingAnnotation, NodeId, Ident, Option<P<Pat>>),

    /// A struct or struct variant pattern, e.g., `Variant {x, y, ..}`.
    /// The `bool` is `true` in the presence of a `..`.
    Struct(QPath, HirVec<Spanned<FieldPat>>, bool),

    /// A tuple struct/variant pattern `Variant(x, y, .., z)`.
    /// If the `..` pattern fragment is present, then `Option<usize>` denotes its position.
    /// 0 <= position <= subpats.len()
    TupleStruct(QPath, HirVec<P<Pat>>, Option<usize>),

    /// A path pattern for an unit struct/variant or a (maybe-associated) constant.
    Path(QPath),

    /// A tuple pattern `(a, b)`.
    /// If the `..` pattern fragment is present, then `Option<usize>` denotes its position.
    /// 0 <= position <= subpats.len()
    Tuple(HirVec<P<Pat>>, Option<usize>),
    /// A `box` pattern
    Box(P<Pat>),
    /// A reference pattern, e.g., `&mut (a, b)`
    Ref(P<Pat>, Mutability),
    /// A literal
    Lit(P<Expr>),
    /// A range pattern, e.g., `1...2` or `1..2`
    Range(P<Expr>, P<Expr>, RangeEnd),
    /// `[a, b, ..i, y, z]` is represented as:
    ///     `PatKind::Slice(box [a, b], Some(i), box [y, z])`
    Slice(HirVec<P<Pat>>, Option<P<Pat>>, HirVec<P<Pat>>),
}

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, RustcEncodable, RustcDecodable, Hash, Debug, Copy)]
pub enum Mutability {
    MutMutable,
    MutImmutable,
}

impl Mutability {
    /// Return MutMutable only if both arguments are mutable.
    pub fn and(self, other: Self) -> Self {
        match self {
            MutMutable => other,
            MutImmutable => MutImmutable,
        }
    }
}

#[derive(Clone, PartialEq, RustcEncodable, RustcDecodable, Debug, Copy, Hash)]
pub enum BinOpKind {
    /// The `+` operator (addition)
    Add,
    /// The `-` operator (subtraction)
    Sub,
    /// The `*` operator (multiplication)
    Mul,
    /// The `/` operator (division)
    Div,
    /// The `%` operator (modulus)
    Rem,
    /// The `&&` operator (logical and)
    And,
    /// The `||` operator (logical or)
    Or,
    /// The `^` operator (bitwise xor)
    BitXor,
    /// The `&` operator (bitwise and)
    BitAnd,
    /// The `|` operator (bitwise or)
    BitOr,
    /// The `<<` operator (shift left)
    Shl,
    /// The `>>` operator (shift right)
    Shr,
    /// The `==` operator (equality)
    Eq,
    /// The `<` operator (less than)
    Lt,
    /// The `<=` operator (less than or equal to)
    Le,
    /// The `!=` operator (not equal to)
    Ne,
    /// The `>=` operator (greater than or equal to)
    Ge,
    /// The `>` operator (greater than)
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

    /// Returns `true` if the binary operator takes its arguments by value
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

#[derive(Clone, PartialEq, RustcEncodable, RustcDecodable, Debug, Copy, Hash)]
pub enum UnOp {
    /// The `*` operator for dereferencing
    UnDeref,
    /// The `!` operator for logical inversion
    UnNot,
    /// The `-` operator for negation
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

    /// Returns `true` if the unary operator takes its argument by value
    pub fn is_by_value(self) -> bool {
        match self {
            UnNeg | UnNot => true,
            _ => false,
        }
    }
}

/// A statement
pub type Stmt = Spanned<StmtKind>;

impl fmt::Debug for StmtKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Sadness.
        let spanned = source_map::dummy_spanned(self.clone());
        write!(f,
               "stmt({}: {})",
               spanned.node.id(),
               print::to_string(print::NO_ANN, |s| s.print_stmt(&spanned)))
    }
}

#[derive(Clone, RustcEncodable, RustcDecodable)]
pub enum StmtKind {
    /// Could be an item or a local (let) binding:
    Decl(P<Decl>, NodeId),

    /// Expr without trailing semi-colon (must have unit type):
    Expr(P<Expr>, NodeId),

    /// Expr with trailing semi-colon (may have any type):
    Semi(P<Expr>, NodeId),
}

impl StmtKind {
    pub fn attrs(&self) -> &[Attribute] {
        match *self {
            StmtKind::Decl(ref d, _) => d.node.attrs(),
            StmtKind::Expr(ref e, _) |
            StmtKind::Semi(ref e, _) => &e.attrs,
        }
    }

    pub fn id(&self) -> NodeId {
        match *self {
            StmtKind::Decl(_, id) |
            StmtKind::Expr(_, id) |
            StmtKind::Semi(_, id) => id,
        }
    }
}

/// Local represents a `let` statement, e.g., `let <pat>:<ty> = <expr>;`
#[derive(Clone, RustcEncodable, RustcDecodable, Debug)]
pub struct Local {
    pub pat: P<Pat>,
    pub ty: Option<P<Ty>>,
    /// Initializer expression to set the value, if any
    pub init: Option<P<Expr>>,
    pub id: NodeId,
    pub hir_id: HirId,
    pub span: Span,
    pub attrs: ThinVec<Attribute>,
    pub source: LocalSource,
}

pub type Decl = Spanned<DeclKind>;

#[derive(Clone, RustcEncodable, RustcDecodable, Debug)]
pub enum DeclKind {
    /// A local (let) binding:
    Local(P<Local>),
    /// An item binding:
    Item(ItemId),
}

impl DeclKind {
    pub fn attrs(&self) -> &[Attribute] {
        match *self {
            DeclKind::Local(ref l) => &l.attrs,
            DeclKind::Item(_) => &[]
        }
    }

    pub fn is_local(&self) -> bool {
        match *self {
            DeclKind::Local(_) => true,
            _ => false,
        }
    }
}

/// represents one arm of a 'match'
#[derive(Clone, RustcEncodable, RustcDecodable, Debug)]
pub struct Arm {
    pub attrs: HirVec<Attribute>,
    pub pats: HirVec<P<Pat>>,
    pub guard: Option<Guard>,
    pub body: P<Expr>,
}

#[derive(Clone, RustcEncodable, RustcDecodable, Debug)]
pub enum Guard {
    If(P<Expr>),
}

#[derive(Clone, RustcEncodable, RustcDecodable, Debug)]
pub struct Field {
    pub id: NodeId,
    pub ident: Ident,
    pub expr: P<Expr>,
    pub span: Span,
    pub is_shorthand: bool,
}

#[derive(Clone, PartialEq, RustcEncodable, RustcDecodable, Debug, Copy)]
pub enum BlockCheckMode {
    DefaultBlock,
    UnsafeBlock(UnsafeSource),
    PushUnsafeBlock(UnsafeSource),
    PopUnsafeBlock(UnsafeSource),
}

#[derive(Clone, PartialEq, RustcEncodable, RustcDecodable, Debug, Copy)]
pub enum UnsafeSource {
    CompilerGenerated,
    UserProvided,
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, RustcEncodable, RustcDecodable, Hash, Debug)]
pub struct BodyId {
    pub node_id: NodeId,
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
/// - `is_generator` would be false
///
/// All bodies have an **owner**, which can be accessed via the HIR
/// map using `body_owner_def_id()`.
#[derive(Clone, RustcEncodable, RustcDecodable, Debug)]
pub struct Body {
    pub arguments: HirVec<Arg>,
    pub value: Expr,
    pub is_generator: bool,
}

impl Body {
    pub fn id(&self) -> BodyId {
        BodyId {
            node_id: self.value.id
        }
    }
}

#[derive(Copy, Clone, Debug)]
pub enum BodyOwnerKind {
    /// Functions and methods.
    Fn,

    /// Constants and associated constants.
    Const,

    /// Initializer of a `static` item.
    Static(Mutability),
}

/// A constant (expression) that's not an item or associated item,
/// but needs its own `DefId` for type-checking, const-eval, etc.
/// These are usually found nested inside types (e.g., array lengths)
/// or expressions (e.g., repeat counts), and also used to define
/// explicit discriminant values for enum variants.
#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Debug)]
pub struct AnonConst {
    pub id: NodeId,
    pub hir_id: HirId,
    pub body: BodyId,
}

/// An expression
#[derive(Clone, RustcEncodable, RustcDecodable)]
pub struct Expr {
    pub id: NodeId,
    pub span: Span,
    pub node: ExprKind,
    pub attrs: ThinVec<Attribute>,
    pub hir_id: HirId,
}

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
            ExprKind::If(..) => ExprPrecedence::If,
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
        }
    }

    pub fn is_place_expr(&self) -> bool {
         match self.node {
            ExprKind::Path(QPath::Resolved(_, ref path)) => {
                match path.def {
                    Def::Local(..) | Def::Upvar(..) | Def::Static(..) | Def::Err => true,
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
            ExprKind::If(..) |
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
            ExprKind::Cast(..) => {
                false
            }
        }
    }
}

impl fmt::Debug for Expr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "expr({}: {})", self.id,
               print::to_string(print::NO_ANN, |s| s.print_expr(self)))
    }
}

#[derive(Clone, RustcEncodable, RustcDecodable, Debug)]
pub enum ExprKind {
    /// A `box x` expression.
    Box(P<Expr>),
    /// An array (`[a, b, c, d]`)
    Array(HirVec<Expr>),
    /// A function call
    ///
    /// The first field resolves to the function itself (usually an `ExprKind::Path`),
    /// and the second field is the list of arguments.
    /// This also represents calling the constructor of
    /// tuple-like ADTs such as tuple structs and enum variants.
    Call(P<Expr>, HirVec<Expr>),
    /// A method call (`x.foo::<'static, Bar, Baz>(a, b, c, d)`)
    ///
    /// The `PathSegment`/`Span` represent the method name and its generic arguments
    /// (within the angle brackets).
    /// The first element of the vector of `Expr`s is the expression that evaluates
    /// to the object on which the method is being called on (the receiver),
    /// and the remaining elements are the rest of the arguments.
    /// Thus, `x.foo::<Bar, Baz>(a, b, c, d)` is represented as
    /// `ExprKind::MethodCall(PathSegment { foo, [Bar, Baz] }, [x, a, b, c, d])`.
    MethodCall(PathSegment, Span, HirVec<Expr>),
    /// A tuple (`(a, b, c ,d)`)
    Tup(HirVec<Expr>),
    /// A binary operation (For example: `a + b`, `a * b`)
    Binary(BinOp, P<Expr>, P<Expr>),
    /// A unary operation (For example: `!x`, `*x`)
    Unary(UnOp, P<Expr>),
    /// A literal (For example: `1`, `"foo"`)
    Lit(P<Lit>),
    /// A cast (`foo as f64`)
    Cast(P<Expr>, P<Ty>),
    Type(P<Expr>, P<Ty>),
    /// An `if` block, with an optional else block
    ///
    /// `if expr { expr } else { expr }`
    If(P<Expr>, P<Expr>, Option<P<Expr>>),
    /// A while loop, with an optional label
    ///
    /// `'label: while expr { block }`
    While(P<Expr>, P<Block>, Option<Label>),
    /// Conditionless loop (can be exited with break, continue, or return)
    ///
    /// `'label: loop { block }`
    Loop(P<Block>, Option<Label>, LoopSource),
    /// A `match` block, with a source that indicates whether or not it is
    /// the result of a desugaring, and if so, which kind.
    Match(P<Expr>, HirVec<Arm>, MatchSource),
    /// A closure (for example, `move |a, b, c| {a + b + c}`).
    ///
    /// The final span is the span of the argument block `|...|`
    ///
    /// This may also be a generator literal, indicated by the final boolean,
    /// in that case there is an GeneratorClause.
    Closure(CaptureClause, P<FnDecl>, BodyId, Span, Option<GeneratorMovability>),
    /// A block (`'label: { ... }`)
    Block(P<Block>, Option<Label>),

    /// An assignment (`a = foo()`)
    Assign(P<Expr>, P<Expr>),
    /// An assignment with an operator
    ///
    /// For example, `a += 1`.
    AssignOp(BinOp, P<Expr>, P<Expr>),
    /// Access of a named (`obj.foo`) or unnamed (`obj.0`) struct or tuple field
    Field(P<Expr>, Ident),
    /// An indexing operation (`foo[2]`)
    Index(P<Expr>, P<Expr>),

    /// Path to a definition, possibly containing lifetime or type parameters.
    Path(QPath),

    /// A referencing operation (`&a` or `&mut a`)
    AddrOf(Mutability, P<Expr>),
    /// A `break`, with an optional label to break
    Break(Destination, Option<P<Expr>>),
    /// A `continue`, with an optional label
    Continue(Destination),
    /// A `return`, with an optional value to be returned
    Ret(Option<P<Expr>>),

    /// Inline assembly (from `asm!`), with its outputs and inputs.
    InlineAsm(P<InlineAsm>, HirVec<Expr>, HirVec<Expr>),

    /// A struct or struct-like variant literal expression.
    ///
    /// For example, `Foo {x: 1, y: 2}`, or
    /// `Foo {x: 1, .. base}`, where `base` is the `Option<Expr>`.
    Struct(QPath, HirVec<Field>, Option<P<Expr>>),

    /// An array literal constructed from one repeated element.
    ///
    /// For example, `[1; 5]`. The first expression is the element
    /// to be repeated; the second is the number of times to repeat it.
    Repeat(P<Expr>, AnonConst),

    /// A suspension point for generators. This is `yield <expr>` in Rust.
    Yield(P<Expr>),
}

/// Optionally `Self`-qualified value/type path or associated extension.
#[derive(Clone, RustcEncodable, RustcDecodable, Debug)]
pub enum QPath {
    /// Path to a definition, optionally "fully-qualified" with a `Self`
    /// type, if the path points to an associated item in a trait.
    ///
    /// e.g., an unqualified path like `Clone::clone` has `None` for `Self`,
    /// while `<Vec<T> as Clone>::clone` has `Some(Vec<T>)` for `Self`,
    /// even though they both have the same two-segment `Clone::clone` `Path`.
    Resolved(Option<P<Ty>>, P<Path>),

    /// Type-related paths, e.g., `<T>::default` or `<T>::Output`.
    /// Will be resolved by type-checking to an associated item.
    ///
    /// UFCS source paths can desugar into this, with `Vec::new` turning into
    /// `<Vec>::new`, and `T::X::Y::method` into `<<<T>::X>::Y>::method`,
    /// the `X` and `Y` nodes each being a `TyKind::Path(QPath::TypeRelative(..))`.
    TypeRelative(P<Ty>, P<PathSegment>)
}

/// Hints at the original code for a let statement
#[derive(Clone, RustcEncodable, RustcDecodable, Debug, Copy)]
pub enum LocalSource {
    /// A `match _ { .. }`
    Normal,
    /// A desugared `for _ in _ { .. }` loop
    ForLoopDesugar,
}

/// Hints at the original code for a `match _ { .. }`
#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Debug, Copy)]
pub enum MatchSource {
    /// A `match _ { .. }`
    Normal,
    /// An `if let _ = _ { .. }` (optionally with `else { .. }`)
    IfLetDesugar {
        contains_else_clause: bool,
    },
    /// A `while let _ = _ { .. }` (which was desugared to a
    /// `loop { match _ { .. } }`)
    WhileLetDesugar,
    /// A desugared `for _ in _ { .. }` loop
    ForLoopDesugar,
    /// A desugared `?` operator
    TryDesugar,
}

/// The loop type that yielded an ExprKind::Loop
#[derive(Clone, PartialEq, RustcEncodable, RustcDecodable, Debug, Copy)]
pub enum LoopSource {
    /// A `loop { .. }` loop
    Loop,
    /// A `while let _ = _ { .. }` loop
    WhileLet,
    /// A `for _ in _ { .. }` loop
    ForLoop,
}

#[derive(Clone, RustcEncodable, RustcDecodable, Debug, Copy)]
pub enum LoopIdError {
    OutsideLoopScope,
    UnlabeledCfInWhileCondition,
    UnresolvedLabel,
}

impl fmt::Display for LoopIdError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(match *self {
            LoopIdError::OutsideLoopScope => "not inside loop scope",
            LoopIdError::UnlabeledCfInWhileCondition =>
                "unlabeled control flow (break or continue) in while condition",
            LoopIdError::UnresolvedLabel => "label not found",
        }, f)
    }
}

#[derive(Clone, RustcEncodable, RustcDecodable, Debug, Copy)]
pub struct Destination {
    // This is `Some(_)` iff there is an explicit user-specified `label
    pub label: Option<Label>,

    // These errors are caught and then reported during the diagnostics pass in
    // librustc_passes/loops.rs
    pub target_id: Result<NodeId, LoopIdError>,
}

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, RustcEncodable, RustcDecodable, Hash, Debug, Copy)]
pub enum GeneratorMovability {
    Static,
    Movable,
}

#[derive(Clone, RustcEncodable, RustcDecodable, Debug, Copy)]
pub enum CaptureClause {
    CaptureByValue,
    CaptureByRef,
}

// N.B., if you change this, you'll probably want to change the corresponding
// type structure in middle/ty.rs as well.
#[derive(Clone, RustcEncodable, RustcDecodable, Debug)]
pub struct MutTy {
    pub ty: P<Ty>,
    pub mutbl: Mutability,
}

/// Represents a method's signature in a trait declaration or implementation.
#[derive(Clone, RustcEncodable, RustcDecodable, Debug)]
pub struct MethodSig {
    pub header: FnHeader,
    pub decl: P<FnDecl>,
}

// The bodies for items are stored "out of line", in a separate
// hashmap in the `Crate`. Here we just record the node-id of the item
// so it can fetched later.
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, RustcEncodable, RustcDecodable, Debug)]
pub struct TraitItemId {
    pub node_id: NodeId,
}

/// Represents an item declaration within a trait declaration,
/// possibly including a default implementation. A trait item is
/// either required (meaning it doesn't have an implementation, just a
/// signature) or provided (meaning it has a default implementation).
#[derive(Clone, RustcEncodable, RustcDecodable, Debug)]
pub struct TraitItem {
    pub id: NodeId,
    pub ident: Ident,
    pub hir_id: HirId,
    pub attrs: HirVec<Attribute>,
    pub generics: Generics,
    pub node: TraitItemKind,
    pub span: Span,
}

/// A trait method's body (or just argument names).
#[derive(Clone, RustcEncodable, RustcDecodable, Debug)]
pub enum TraitMethod {
    /// No default body in the trait, just a signature.
    Required(HirVec<Ident>),

    /// Both signature and body are provided in the trait.
    Provided(BodyId),
}

/// Represents a trait method or associated constant or type
#[derive(Clone, RustcEncodable, RustcDecodable, Debug)]
pub enum TraitItemKind {
    /// An associated constant with an optional value (otherwise `impl`s
    /// must contain a value)
    Const(P<Ty>, Option<BodyId>),
    /// A method with an optional body
    Method(MethodSig, TraitMethod),
    /// An associated type with (possibly empty) bounds and optional concrete
    /// type
    Type(GenericBounds, Option<P<Ty>>),
}

// The bodies for items are stored "out of line", in a separate
// hashmap in the `Crate`. Here we just record the node-id of the item
// so it can fetched later.
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, RustcEncodable, RustcDecodable, Debug)]
pub struct ImplItemId {
    pub node_id: NodeId,
}

/// Represents anything within an `impl` block
#[derive(Clone, RustcEncodable, RustcDecodable, Debug)]
pub struct ImplItem {
    pub id: NodeId,
    pub ident: Ident,
    pub hir_id: HirId,
    pub vis: Visibility,
    pub defaultness: Defaultness,
    pub attrs: HirVec<Attribute>,
    pub generics: Generics,
    pub node: ImplItemKind,
    pub span: Span,
}

/// Represents different contents within `impl`s
#[derive(Clone, RustcEncodable, RustcDecodable, Debug)]
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

// Bind a type to an associated type: `A=Foo`.
#[derive(Clone, RustcEncodable, RustcDecodable, Debug)]
pub struct TypeBinding {
    pub id: NodeId,
    pub ident: Ident,
    pub ty: P<Ty>,
    pub span: Span,
}

#[derive(Clone, RustcEncodable, RustcDecodable)]
pub struct Ty {
    pub id: NodeId,
    pub node: TyKind,
    pub span: Span,
    pub hir_id: HirId,
}

impl fmt::Debug for Ty {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "type({})",
               print::to_string(print::NO_ANN, |s| s.print_type(self)))
    }
}

/// Not represented directly in the AST, referred to by name through a ty_path.
#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Debug, Copy)]
pub enum PrimTy {
    Int(IntTy),
    Uint(UintTy),
    Float(FloatTy),
    Str,
    Bool,
    Char,
}

#[derive(Clone, RustcEncodable, RustcDecodable, Debug)]
pub struct BareFnTy {
    pub unsafety: Unsafety,
    pub abi: Abi,
    pub generic_params: HirVec<GenericParam>,
    pub decl: P<FnDecl>,
    pub arg_names: HirVec<Ident>,
}

#[derive(Clone, RustcEncodable, RustcDecodable, Debug)]
pub struct ExistTy {
    pub generics: Generics,
    pub bounds: GenericBounds,
    pub impl_trait_fn: Option<DefId>,
}

#[derive(Clone, RustcEncodable, RustcDecodable, Debug)]
/// The different kinds of types recognized by the compiler
pub enum TyKind {
    /// A variable length slice (`[T]`)
    Slice(P<Ty>),
    /// A fixed length array (`[T; n]`)
    Array(P<Ty>, AnonConst),
    /// A raw pointer (`*const T` or `*mut T`)
    Ptr(MutTy),
    /// A reference (`&'a T` or `&'a mut T`)
    Rptr(Lifetime, MutTy),
    /// A bare function (e.g., `fn(usize) -> bool`)
    BareFn(P<BareFnTy>),
    /// The never type (`!`)
    Never,
    /// A tuple (`(A, B, C, D,...)`)
    Tup(HirVec<Ty>),
    /// A path to a type definition (`module::module::...::Type`), or an
    /// associated type, e.g., `<Vec<T> as Trait>::Type` or `<T>::Target`.
    ///
    /// Type parameters may be stored in each `PathSegment`.
    Path(QPath),
    /// A type definition itself. This is currently only used for the `existential type`
    /// item that `impl Trait` in return position desugars to.
    ///
    /// The generic arg list are the lifetimes (and in the future possibly parameters) that are
    /// actually bound on the `impl Trait`.
    Def(ItemId, HirVec<GenericArg>),
    /// A trait object type `Bound1 + Bound2 + Bound3`
    /// where `Bound` is a trait or a lifetime.
    TraitObject(HirVec<PolyTraitRef>, Lifetime),
    /// Unused for now
    Typeof(AnonConst),
    /// `TyKind::Infer` means the type should be inferred instead of it having been
    /// specified. This can appear anywhere in a type.
    Infer,
    /// Placeholder for a type that has failed to be defined.
    Err,
}

#[derive(Clone, RustcEncodable, RustcDecodable, Debug)]
pub struct InlineAsmOutput {
    pub constraint: Symbol,
    pub is_rw: bool,
    pub is_indirect: bool,
    pub span: Span,
}

#[derive(Clone, RustcEncodable, RustcDecodable, Debug)]
pub struct InlineAsm {
    pub asm: Symbol,
    pub asm_str_style: StrStyle,
    pub outputs: HirVec<InlineAsmOutput>,
    pub inputs: HirVec<Symbol>,
    pub clobbers: HirVec<Symbol>,
    pub volatile: bool,
    pub alignstack: bool,
    pub dialect: AsmDialect,
    pub ctxt: SyntaxContext,
}

/// represents an argument in a function header
#[derive(Clone, RustcEncodable, RustcDecodable, Debug)]
pub struct Arg {
    pub pat: P<Pat>,
    pub id: NodeId,
    pub hir_id: HirId,
}

/// Represents the header (not the body) of a function declaration
#[derive(Clone, RustcEncodable, RustcDecodable, Debug)]
pub struct FnDecl {
    pub inputs: HirVec<Ty>,
    pub output: FunctionRetTy,
    pub variadic: bool,
    /// Does the function have an implicit self?
    pub implicit_self: ImplicitSelfKind,
}

/// Represents what type of implicit self a function has, if any.
#[derive(Clone, Copy, RustcEncodable, RustcDecodable, Debug)]
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
#[derive(Copy, Clone, PartialEq, RustcEncodable, RustcDecodable, Debug)]
pub enum IsAuto {
    Yes,
    No
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, RustcEncodable, RustcDecodable, Debug)]
pub enum IsAsync {
    Async,
    NotAsync,
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, RustcEncodable, RustcDecodable, Hash, Debug)]
pub enum Unsafety {
    Unsafe,
    Normal,
}

#[derive(Copy, Clone, PartialEq, RustcEncodable, RustcDecodable, Debug)]
pub enum Constness {
    Const,
    NotConst,
}

#[derive(Copy, Clone, PartialEq, RustcEncodable, RustcDecodable, Debug)]
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
        fmt::Display::fmt(match *self {
                              Unsafety::Normal => "normal",
                              Unsafety::Unsafe => "unsafe",
                          },
                          f)
    }
}

#[derive(Copy, Clone, PartialEq, RustcEncodable, RustcDecodable)]
pub enum ImplPolarity {
    /// `impl Trait for Type`
    Positive,
    /// `impl !Trait for Type`
    Negative,
}

impl fmt::Debug for ImplPolarity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            ImplPolarity::Positive => "positive".fmt(f),
            ImplPolarity::Negative => "negative".fmt(f),
        }
    }
}


#[derive(Clone, RustcEncodable, RustcDecodable, Debug)]
pub enum FunctionRetTy {
    /// Return type is not specified.
    ///
    /// Functions default to `()` and
    /// closures default to inference. Span points to where return
    /// type would be inserted.
    DefaultReturn(Span),
    /// Everything else
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

#[derive(Clone, RustcEncodable, RustcDecodable, Debug)]
pub struct Mod {
    /// A span from the first token past `{` to the last token until `}`.
    /// For `mod foo;`, the inner span ranges from the first token
    /// to the last token in the external file.
    pub inner: Span,
    pub item_ids: HirVec<ItemId>,
}

#[derive(Clone, RustcEncodable, RustcDecodable, Debug)]
pub struct ForeignMod {
    pub abi: Abi,
    pub items: HirVec<ForeignItem>,
}

#[derive(Clone, RustcEncodable, RustcDecodable, Debug)]
pub struct GlobalAsm {
    pub asm: Symbol,
    pub ctxt: SyntaxContext,
}

#[derive(Clone, RustcEncodable, RustcDecodable, Debug)]
pub struct EnumDef {
    pub variants: HirVec<Variant>,
}

#[derive(Clone, RustcEncodable, RustcDecodable, Debug)]
pub struct VariantKind {
    pub name: Name,
    pub attrs: HirVec<Attribute>,
    pub data: VariantData,
    /// Explicit discriminant, e.g., `Foo = 1`
    pub disr_expr: Option<AnonConst>,
}

pub type Variant = Spanned<VariantKind>;

#[derive(Copy, Clone, PartialEq, RustcEncodable, RustcDecodable, Debug)]
pub enum UseKind {
    /// One import, e.g., `use foo::bar` or `use foo::bar as baz`.
    /// Also produced for each element of a list `use`, e.g.
    // `use foo::{a, b}` lowers to `use foo::a; use foo::b;`.
    Single,

    /// Glob import, e.g., `use foo::*`.
    Glob,

    /// Degenerate list import, e.g., `use foo::{a, b}` produces
    /// an additional `use foo::{}` for performing checks such as
    /// unstable feature gating. May be removed in the future.
    ListStem,
}

/// TraitRef's appear in impls.
///
/// resolve maps each TraitRef's ref_id to its defining trait; that's all
/// that the ref_id is for. Note that ref_id's value is not the NodeId of the
/// trait being referred to but just a unique NodeId that serves as a key
/// within the DefMap.
#[derive(Clone, RustcEncodable, RustcDecodable, Debug)]
pub struct TraitRef {
    pub path: Path,
    pub ref_id: NodeId,
    pub hir_ref_id: HirId,
}

#[derive(Clone, RustcEncodable, RustcDecodable, Debug)]
pub struct PolyTraitRef {
    /// The `'a` in `<'a> Foo<&'a T>`
    pub bound_generic_params: HirVec<GenericParam>,

    /// The `Foo<&'a T>` in `<'a> Foo<&'a T>`
    pub trait_ref: TraitRef,

    pub span: Span,
}

pub type Visibility = Spanned<VisibilityKind>;

#[derive(Clone, RustcEncodable, RustcDecodable, Debug)]
pub enum VisibilityKind {
    Public,
    Crate(CrateSugar),
    Restricted { path: P<Path>, id: NodeId, hir_id: HirId },
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
}

#[derive(Clone, RustcEncodable, RustcDecodable, Debug)]
pub struct StructField {
    pub span: Span,
    pub ident: Ident,
    pub vis: Visibility,
    pub id: NodeId,
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

/// Fields and Ids of enum variants and structs
///
/// For enum variants: `NodeId` represents both an Id of the variant itself (relevant for all
/// variant kinds) and an Id of the variant's constructor (not relevant for `Struct`-variants).
/// One shared Id can be successfully used for these two purposes.
/// Id of the whole enum lives in `Item`.
///
/// For structs: `NodeId` represents an Id of the structure's constructor, so it is not actually
/// used for `Struct`-structs (but still present). Structures don't have an analogue of "Id of
/// the variant itself" from enum variants.
/// Id of the whole struct lives in `Item`.
#[derive(Clone, RustcEncodable, RustcDecodable, Debug)]
pub enum VariantData {
    Struct(HirVec<StructField>, NodeId),
    Tuple(HirVec<StructField>, NodeId),
    Unit(NodeId),
}

impl VariantData {
    pub fn fields(&self) -> &[StructField] {
        match *self {
            VariantData::Struct(ref fields, _) | VariantData::Tuple(ref fields, _) => fields,
            _ => &[],
        }
    }
    pub fn id(&self) -> NodeId {
        match *self {
            VariantData::Struct(_, id) | VariantData::Tuple(_, id) | VariantData::Unit(id) => id,
        }
    }
    pub fn is_struct(&self) -> bool {
        if let VariantData::Struct(..) = *self {
            true
        } else {
            false
        }
    }
    pub fn is_tuple(&self) -> bool {
        if let VariantData::Tuple(..) = *self {
            true
        } else {
            false
        }
    }
    pub fn is_unit(&self) -> bool {
        if let VariantData::Unit(..) = *self {
            true
        } else {
            false
        }
    }
}

// The bodies for items are stored "out of line", in a separate
// hashmap in the `Crate`. Here we just record the node-id of the item
// so it can fetched later.
#[derive(Copy, Clone, RustcEncodable, RustcDecodable, Debug)]
pub struct ItemId {
    pub id: NodeId,
}

/// An item
///
/// The name might be a dummy name in case of anonymous items
#[derive(Clone, RustcEncodable, RustcDecodable, Debug)]
pub struct Item {
    pub name: Name,
    pub id: NodeId,
    pub hir_id: HirId,
    pub attrs: HirVec<Attribute>,
    pub node: ItemKind,
    pub vis: Visibility,
    pub span: Span,
}

#[derive(Clone, Copy, RustcEncodable, RustcDecodable, Debug)]
pub struct FnHeader {
    pub unsafety: Unsafety,
    pub constness: Constness,
    pub asyncness: IsAsync,
    pub abi: Abi,
}

#[derive(Clone, RustcEncodable, RustcDecodable, Debug)]
pub enum ItemKind {
    /// An `extern crate` item, with optional *original* crate name if the crate was renamed.
    ///
    /// e.g., `extern crate foo` or `extern crate foo_bar as foo`
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
            ItemKind::Impl(..) => "item",
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
/// passes to find the impl they want without loading the id (which
/// means fewer edges in the incremental compilation graph).
#[derive(Clone, RustcEncodable, RustcDecodable, Debug)]
pub struct TraitItemRef {
    pub id: TraitItemId,
    pub ident: Ident,
    pub kind: AssociatedItemKind,
    pub span: Span,
    pub defaultness: Defaultness,
}

/// A reference from an impl to one of its associated items. This
/// contains the item's id, naturally, but also the item's name and
/// some other high-level details (like whether it is an associated
/// type or method, and whether it is public). This allows other
/// passes to find the impl they want without loading the id (which
/// means fewer edges in the incremental compilation graph).
#[derive(Clone, RustcEncodable, RustcDecodable, Debug)]
pub struct ImplItemRef {
    pub id: ImplItemId,
    pub ident: Ident,
    pub kind: AssociatedItemKind,
    pub span: Span,
    pub vis: Visibility,
    pub defaultness: Defaultness,
}

#[derive(Copy, Clone, PartialEq, RustcEncodable, RustcDecodable, Debug)]
pub enum AssociatedItemKind {
    Const,
    Method { has_self: bool },
    Type,
    Existential,
}

#[derive(Clone, RustcEncodable, RustcDecodable, Debug)]
pub struct ForeignItem {
    pub name: Name,
    pub attrs: HirVec<Attribute>,
    pub node: ForeignItemKind,
    pub id: NodeId,
    pub span: Span,
    pub vis: Visibility,
}

/// An item within an `extern` block
#[derive(Clone, RustcEncodable, RustcDecodable, Debug)]
pub enum ForeignItemKind {
    /// A foreign function
    Fn(P<FnDecl>, HirVec<Ident>, Generics),
    /// A foreign static item (`static ext: u8`), with optional mutability
    /// (the boolean is true when mutable)
    Static(P<Ty>, bool),
    /// A foreign type
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

/// A free variable referred to in a function.
#[derive(Debug, Copy, Clone, RustcEncodable, RustcDecodable)]
pub struct Freevar {
    /// The variable being accessed free.
    pub def: Def,

    // First span where it is accessed (there can be multiple).
    pub span: Span
}

impl Freevar {
    pub fn var_id(&self) -> NodeId {
        match self.def {
            Def::Local(id) | Def::Upvar(id, ..) => id,
            _ => bug!("Freevar::var_id: bad def ({:?})", self.def)
        }
    }
}

pub type FreevarMap = NodeMap<Vec<Freevar>>;

pub type CaptureModeMap = NodeMap<CaptureClause>;

#[derive(Clone, Debug)]
pub struct TraitCandidate {
    pub def_id: DefId,
    pub import_id: Option<NodeId>,
}

// Trait method resolution
pub type TraitMap = NodeMap<Vec<TraitCandidate>>;

// Map from the NodeId of a glob import to a list of items which are actually
// imported.
pub type GlobMap = NodeMap<FxHashSet<Name>>;


pub fn provide(providers: &mut Providers<'_>) {
    providers.describe_def = map::describe_def;
}

#[derive(Clone, RustcEncodable, RustcDecodable)]
pub struct CodegenFnAttrs {
    pub flags: CodegenFnAttrFlags,
    /// Parsed representation of the `#[inline]` attribute
    pub inline: InlineAttr,
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
    #[derive(RustcEncodable, RustcDecodable)]
    pub struct CodegenFnAttrFlags: u32 {
        /// #[cold], a hint to LLVM that this function, when called, is never on
        /// the hot path
        const COLD                      = 1 << 0;
        /// #[allocator], a hint to LLVM that the pointer returned from this
        /// function is never null
        const ALLOCATOR                 = 1 << 1;
        /// #[unwind], an indicator that this function may unwind despite what
        /// its ABI signature may otherwise imply
        const UNWIND                    = 1 << 2;
        /// #[rust_allocator_nounwind], an indicator that an imported FFI
        /// function will never unwind. Probably obsolete by recent changes with
        /// #[unwind], but hasn't been removed/migrated yet
        const RUSTC_ALLOCATOR_NOUNWIND  = 1 << 3;
        /// #[naked], indicates to LLVM that no function prologue/epilogue
        /// should be generated
        const NAKED                     = 1 << 4;
        /// #[no_mangle], the function's name should be the same as its symbol
        const NO_MANGLE                 = 1 << 5;
        /// #[rustc_std_internal_symbol], and indicator that this symbol is a
        /// "weird symbol" for the standard library in that it has slightly
        /// different linkage, visibility, and reachability rules.
        const RUSTC_STD_INTERNAL_SYMBOL = 1 << 6;
        /// #[no_debug], indicates that no debugging information should be
        /// generated for this function by LLVM
        const NO_DEBUG                  = 1 << 7;
        /// #[thread_local], indicates a static is actually a thread local
        /// piece of memory
        const THREAD_LOCAL              = 1 << 8;
        /// #[used], indicates that LLVM can't eliminate this function (but the
        /// linker can!)
        const USED                      = 1 << 9;
    }
}

impl CodegenFnAttrs {
    pub fn new() -> CodegenFnAttrs {
        CodegenFnAttrs {
            flags: CodegenFnAttrFlags::empty(),
            inline: InlineAttr::None,
            export_name: None,
            link_name: None,
            target_features: vec![],
            linkage: None,
            link_section: None,
        }
    }

    /// True if `#[inline]` or `#[inline(always)]` is present.
    pub fn requests_inline(&self) -> bool {
        match self.inline {
            InlineAttr::Hint | InlineAttr::Always => true,
            InlineAttr::None | InlineAttr::Never => false,
        }
    }

    /// True if it looks like this symbol needs to be exported, for example:
    ///
    /// * `#[no_mangle]` is present
    /// * `#[export_name(...)]` is present
    /// * `#[linkage]` is present
    pub fn contains_extern_indicator(&self) -> bool {
        self.flags.contains(CodegenFnAttrFlags::NO_MANGLE) ||
            self.export_name.is_some() ||
            match self.linkage {
                // these are private, make sure we don't try to consider
                // them external
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
    Block(&'hir Block),
    Local(&'hir Local),
    MacroDef(&'hir MacroDef),

    /// StructCtor represents a tuple struct.
    StructCtor(&'hir VariantData),

    Lifetime(&'hir Lifetime),
    GenericParam(&'hir GenericParam),
    Visibility(&'hir Visibility),

    Crate,
}
