//! **Canonicalization** is the key to constructing a query in the
//! middle of type inference. Ordinarily, it is not possible to store
//! types from type inference in query keys, because they contain
//! references to inference variables whose lifetimes are too short
//! and so forth. Canonicalizing a value T1 using `canonicalize_query`
//! produces two things:
//!
//! - a value T2 where each unbound inference variable has been
//!   replaced with a **canonical variable**;
//! - a map M (of type `CanonicalVarValues`) from those canonical
//!   variables back to the original.
//!
//! We can then do queries using T2. These will give back constraints
//! on the canonical variables which can be translated, using the map
//! M, into constraints in our source context. This process of
//! translating the results back is done by the
//! `instantiate_query_result` method.
//!
//! For a more detailed look at what is happening here, check
//! out the [chapter in the rustc dev guide][c].
//!
//! [c]: https://rust-lang.github.io/chalk/book/canonical_queries/canonicalization.html

use crate::infer::MemberConstraint;
use crate::ty::subst::GenericArg;
use crate::ty::{self, BoundVar, List, Region, TyCtxt};
use rustc_index::vec::IndexVec;
use rustc_macros::HashStable;
use smallvec::SmallVec;
use std::ops::Index;

/// A "canonicalized" type `V` is one where all free inference
/// variables have been rewritten to "canonical vars". These are
/// numbered starting from 0 in order of first appearance.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, TyDecodable, TyEncodable)]
#[derive(HashStable, TypeFoldable, Lift)]
pub struct Canonical<'tcx, V> {
    pub max_universe: ty::UniverseIndex,
    pub variables: CanonicalVarInfos<'tcx>,
    pub value: V,
}

pub type CanonicalVarInfos<'tcx> = &'tcx List<CanonicalVarInfo<'tcx>>;

/// A set of values corresponding to the canonical variables from some
/// `Canonical`. You can give these values to
/// `canonical_value.substitute` to substitute them into the canonical
/// value at the right places.
///
/// When you canonicalize a value `V`, you get back one of these
/// vectors with the original values that were replaced by canonical
/// variables. You will need to supply it later to instantiate the
/// canonicalized query response.
#[derive(Clone, Debug, PartialEq, Eq, Hash, TyDecodable, TyEncodable)]
#[derive(HashStable, TypeFoldable, Lift)]
pub struct CanonicalVarValues<'tcx> {
    pub var_values: IndexVec<BoundVar, GenericArg<'tcx>>,
}

/// When we canonicalize a value to form a query, we wind up replacing
/// various parts of it with canonical variables. This struct stores
/// those replaced bits to remember for when we process the query
/// result.
#[derive(Clone, Debug)]
pub struct OriginalQueryValues<'tcx> {
    /// Map from the universes that appear in the query to the
    /// universes in the caller context. For the time being, we only
    /// ever put ROOT values into the query, so this map is very
    /// simple.
    pub universe_map: SmallVec<[ty::UniverseIndex; 4]>,

    /// This is equivalent to `CanonicalVarValues`, but using a
    /// `SmallVec` yields a significant performance win.
    pub var_values: SmallVec<[GenericArg<'tcx>; 8]>,
}

impl Default for OriginalQueryValues<'tcx> {
    fn default() -> Self {
        let mut universe_map = SmallVec::default();
        universe_map.push(ty::UniverseIndex::ROOT);

        Self { universe_map, var_values: SmallVec::default() }
    }
}

/// Information about a canonical variable that is included with the
/// canonical value. This is sufficient information for code to create
/// a copy of the canonical value in some other inference context,
/// with fresh inference variables replacing the canonical values.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, TyDecodable, TyEncodable, HashStable)]
pub struct CanonicalVarInfo<'tcx> {
    pub kind: CanonicalVarKind<'tcx>,
}

impl<'tcx> CanonicalVarInfo<'tcx> {
    pub fn universe(&self) -> ty::UniverseIndex {
        self.kind.universe()
    }

    pub fn is_existential(&self) -> bool {
        match self.kind {
            CanonicalVarKind::Ty(_) => true,
            CanonicalVarKind::PlaceholderTy(_) => false,
            CanonicalVarKind::Region(_) => true,
            CanonicalVarKind::PlaceholderRegion(..) => false,
            CanonicalVarKind::Const(_) => true,
            CanonicalVarKind::PlaceholderConst(_) => false,
        }
    }
}

/// Describes the "kind" of the canonical variable. This is a "kind"
/// in the type-theory sense of the term -- i.e., a "meta" type system
/// that analyzes type-like values.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, TyDecodable, TyEncodable, HashStable)]
pub enum CanonicalVarKind<'tcx> {
    /// Some kind of type inference variable.
    Ty(CanonicalTyVarKind),

    /// A "placeholder" that represents "any type".
    PlaceholderTy(ty::PlaceholderType),

    /// Region variable `'?R`.
    Region(ty::UniverseIndex),

    /// A "placeholder" that represents "any region". Created when you
    /// are solving a goal like `for<'a> T: Foo<'a>` to represent the
    /// bound region `'a`.
    PlaceholderRegion(ty::PlaceholderRegion),

    /// Some kind of const inference variable.
    Const(ty::UniverseIndex),

    /// A "placeholder" that represents "any const".
    PlaceholderConst(ty::PlaceholderConst<'tcx>),
}

impl<'tcx> CanonicalVarKind<'tcx> {
    pub fn universe(self) -> ty::UniverseIndex {
        match self {
            CanonicalVarKind::Ty(kind) => match kind {
                CanonicalTyVarKind::General(ui) => ui,
                CanonicalTyVarKind::Float | CanonicalTyVarKind::Int => ty::UniverseIndex::ROOT,
            },

            CanonicalVarKind::PlaceholderTy(placeholder) => placeholder.universe,
            CanonicalVarKind::Region(ui) => ui,
            CanonicalVarKind::PlaceholderRegion(placeholder) => placeholder.universe,
            CanonicalVarKind::Const(ui) => ui,
            CanonicalVarKind::PlaceholderConst(placeholder) => placeholder.universe,
        }
    }
}

/// Rust actually has more than one category of type variables;
/// notably, the type variables we create for literals (e.g., 22 or
/// 22.) can only be instantiated with integral/float types (e.g.,
/// usize or f32). In order to faithfully reproduce a type, we need to
/// know what set of types a given type variable can be unified with.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, TyDecodable, TyEncodable, HashStable)]
pub enum CanonicalTyVarKind {
    /// General type variable `?T` that can be unified with arbitrary types.
    General(ty::UniverseIndex),

    /// Integral type variable `?I` (that can only be unified with integral types).
    Int,

    /// Floating-point type variable `?F` (that can only be unified with float types).
    Float,
}

/// After we execute a query with a canonicalized key, we get back a
/// `Canonical<QueryResponse<..>>`. You can use
/// `instantiate_query_result` to access the data in this result.
#[derive(Clone, Debug, HashStable, TypeFoldable, Lift)]
pub struct QueryResponse<'tcx, R> {
    pub var_values: CanonicalVarValues<'tcx>,
    pub region_constraints: QueryRegionConstraints<'tcx>,
    pub certainty: Certainty,
    pub value: R,
}

#[derive(Clone, Debug, Default, HashStable, TypeFoldable, Lift)]
pub struct QueryRegionConstraints<'tcx> {
    pub outlives: Vec<QueryOutlivesConstraint<'tcx>>,
    pub member_constraints: Vec<MemberConstraint<'tcx>>,
}

impl QueryRegionConstraints<'_> {
    /// Represents an empty (trivially true) set of region
    /// constraints.
    pub fn is_empty(&self) -> bool {
        self.outlives.is_empty() && self.member_constraints.is_empty()
    }
}

pub type Canonicalized<'tcx, V> = Canonical<'tcx, V>;

pub type CanonicalizedQueryResponse<'tcx, T> = &'tcx Canonical<'tcx, QueryResponse<'tcx, T>>;

/// Indicates whether or not we were able to prove the query to be
/// true.
#[derive(Copy, Clone, Debug, HashStable)]
pub enum Certainty {
    /// The query is known to be true, presuming that you apply the
    /// given `var_values` and the region-constraints are satisfied.
    Proven,

    /// The query is not known to be true, but also not known to be
    /// false. The `var_values` represent *either* values that must
    /// hold in order for the query to be true, or helpful tips that
    /// *might* make it true. Currently rustc's trait solver cannot
    /// distinguish the two (e.g., due to our preference for where
    /// clauses over impls).
    ///
    /// After some unifiations and things have been done, it makes
    /// sense to try and prove again -- of course, at that point, the
    /// canonical form will be different, making this a distinct
    /// query.
    Ambiguous,
}

impl Certainty {
    pub fn is_proven(&self) -> bool {
        match self {
            Certainty::Proven => true,
            Certainty::Ambiguous => false,
        }
    }

    pub fn is_ambiguous(&self) -> bool {
        !self.is_proven()
    }
}

impl<'tcx, R> QueryResponse<'tcx, R> {
    pub fn is_proven(&self) -> bool {
        self.certainty.is_proven()
    }

    pub fn is_ambiguous(&self) -> bool {
        !self.is_proven()
    }
}

impl<'tcx, R> Canonical<'tcx, QueryResponse<'tcx, R>> {
    pub fn is_proven(&self) -> bool {
        self.value.is_proven()
    }

    pub fn is_ambiguous(&self) -> bool {
        !self.is_proven()
    }
}

impl<'tcx, V> Canonical<'tcx, V> {
    /// Allows you to map the `value` of a canonical while keeping the
    /// same set of bound variables.
    ///
    /// **WARNING:** This function is very easy to mis-use, hence the
    /// name!  In particular, the new value `W` must use all **the
    /// same type/region variables** in **precisely the same order**
    /// as the original! (The ordering is defined by the
    /// `TypeFoldable` implementation of the type in question.)
    ///
    /// An example of a **correct** use of this:
    ///
    /// ```rust,ignore (not real code)
    /// let a: Canonical<'_, T> = ...;
    /// let b: Canonical<'_, (T,)> = a.unchecked_map(|v| (v, ));
    /// ```
    ///
    /// An example of an **incorrect** use of this:
    ///
    /// ```rust,ignore (not real code)
    /// let a: Canonical<'tcx, T> = ...;
    /// let ty: Ty<'tcx> = ...;
    /// let b: Canonical<'tcx, (T, Ty<'tcx>)> = a.unchecked_map(|v| (v, ty));
    /// ```
    pub fn unchecked_map<W>(self, map_op: impl FnOnce(V) -> W) -> Canonical<'tcx, W> {
        let Canonical { max_universe, variables, value } = self;
        Canonical { max_universe, variables, value: map_op(value) }
    }
}

pub type QueryOutlivesConstraint<'tcx> =
    ty::Binder<ty::OutlivesPredicate<GenericArg<'tcx>, Region<'tcx>>>;

CloneTypeFoldableAndLiftImpls! {
    for <'tcx> {
        crate::infer::canonical::Certainty,
        crate::infer::canonical::CanonicalVarInfo<'tcx>,
        crate::infer::canonical::CanonicalVarKind<'tcx>,
    }
}

CloneTypeFoldableImpls! {
    for <'tcx> {
        crate::infer::canonical::CanonicalVarInfos<'tcx>,
    }
}

impl<'tcx> CanonicalVarValues<'tcx> {
    pub fn len(&self) -> usize {
        self.var_values.len()
    }

    /// Makes an identity substitution from this one: each bound var
    /// is matched to the same bound var, preserving the original kinds.
    /// For example, if we have:
    /// `self.var_values == [Type(u32), Lifetime('a), Type(u64)]`
    /// we'll return a substitution `subst` with:
    /// `subst.var_values == [Type(^0), Lifetime(^1), Type(^2)]`.
    pub fn make_identity(&self, tcx: TyCtxt<'tcx>) -> Self {
        use crate::ty::subst::GenericArgKind;

        CanonicalVarValues {
            var_values: self
                .var_values
                .iter()
                .zip(0..)
                .map(|(kind, i)| match kind.unpack() {
                    GenericArgKind::Type(..) => {
                        tcx.mk_ty(ty::Bound(ty::INNERMOST, ty::BoundVar::from_u32(i).into())).into()
                    }
                    GenericArgKind::Lifetime(..) => tcx
                        .mk_region(ty::ReLateBound(ty::INNERMOST, ty::BoundRegion::BrAnon(i)))
                        .into(),
                    GenericArgKind::Const(ct) => tcx
                        .mk_const(ty::Const {
                            ty: ct.ty,
                            val: ty::ConstKind::Bound(ty::INNERMOST, ty::BoundVar::from_u32(i)),
                        })
                        .into(),
                })
                .collect(),
        }
    }
}

impl<'a, 'tcx> IntoIterator for &'a CanonicalVarValues<'tcx> {
    type Item = GenericArg<'tcx>;
    type IntoIter = ::std::iter::Cloned<::std::slice::Iter<'a, GenericArg<'tcx>>>;

    fn into_iter(self) -> Self::IntoIter {
        self.var_values.iter().cloned()
    }
}

impl<'tcx> Index<BoundVar> for CanonicalVarValues<'tcx> {
    type Output = GenericArg<'tcx>;

    fn index(&self, value: BoundVar) -> &GenericArg<'tcx> {
        &self.var_values[value]
    }
}
