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
use crate::mir::ConstraintCategory;
use crate::ty::subst::GenericArg;
use crate::ty::{self, BoundVar, List, Region, Ty, TyCtxt};
use rustc_macros::HashStable;
use smallvec::SmallVec;
use std::ops::Index;

/// A "canonicalized" type `V` is one where all free inference
/// variables have been rewritten to "canonical vars". These are
/// numbered starting from 0 in order of first appearance.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, TyDecodable, TyEncodable)]
#[derive(HashStable, TypeFoldable, TypeVisitable, Lift)]
pub struct Canonical<'tcx, V> {
    pub value: V,
    pub max_universe: ty::UniverseIndex,
    pub variables: CanonicalVarInfos<'tcx>,
}

pub type CanonicalVarInfos<'tcx> = &'tcx List<CanonicalVarInfo<'tcx>>;

impl<'tcx> ty::TypeFoldable<TyCtxt<'tcx>> for CanonicalVarInfos<'tcx> {
    fn try_fold_with<F: ty::FallibleTypeFolder<TyCtxt<'tcx>>>(
        self,
        folder: &mut F,
    ) -> Result<Self, F::Error> {
        ty::util::fold_list(self, folder, |tcx, v| tcx.mk_canonical_var_infos(v))
    }
}

/// A set of values corresponding to the canonical variables from some
/// `Canonical`. You can give these values to
/// `canonical_value.substitute` to substitute them into the canonical
/// value at the right places.
///
/// When you canonicalize a value `V`, you get back one of these
/// vectors with the original values that were replaced by canonical
/// variables. You will need to supply it later to instantiate the
/// canonicalized query response.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, TyDecodable, TyEncodable)]
#[derive(HashStable, TypeFoldable, TypeVisitable, Lift)]
pub struct CanonicalVarValues<'tcx> {
    pub var_values: ty::SubstsRef<'tcx>,
}

impl CanonicalVarValues<'_> {
    pub fn is_identity(&self) -> bool {
        self.var_values.iter().enumerate().all(|(bv, arg)| match arg.unpack() {
            ty::GenericArgKind::Lifetime(r) => {
                matches!(*r, ty::ReLateBound(ty::INNERMOST, br) if br.var.as_usize() == bv)
            }
            ty::GenericArgKind::Type(ty) => {
                matches!(*ty.kind(), ty::Bound(ty::INNERMOST, bt) if bt.var.as_usize() == bv)
            }
            ty::GenericArgKind::Const(ct) => {
                matches!(ct.kind(), ty::ConstKind::Bound(ty::INNERMOST, bc) if bc.as_usize() == bv)
            }
        })
    }

    pub fn is_identity_modulo_regions(&self) -> bool {
        let mut var = ty::BoundVar::from_u32(0);
        for arg in self.var_values {
            match arg.unpack() {
                ty::GenericArgKind::Lifetime(r) => {
                    if let ty::ReLateBound(ty::INNERMOST, br) = *r
                        && var == br.var
                    {
                        var = var + 1;
                    } else {
                        // It's ok if this region var isn't unique
                    }
                },
                ty::GenericArgKind::Type(ty) => {
                    if let ty::Bound(ty::INNERMOST, bt) = *ty.kind()
                        && var == bt.var
                    {
                        var = var + 1;
                    } else {
                        return false;
                    }
                }
                ty::GenericArgKind::Const(ct) => {
                    if let ty::ConstKind::Bound(ty::INNERMOST, bc) = ct.kind()
                        && var == bc
                    {
                        var = var + 1;
                    } else {
                        return false;
                    }
                }
            }
        }

        true
    }
}

/// When we canonicalize a value to form a query, we wind up replacing
/// various parts of it with canonical variables. This struct stores
/// those replaced bits to remember for when we process the query
/// result.
#[derive(Clone, Debug)]
pub struct OriginalQueryValues<'tcx> {
    /// Map from the universes that appear in the query to the universes in the
    /// caller context. For all queries except `evaluate_goal` (used by Chalk),
    /// we only ever put ROOT values into the query, so this map is very
    /// simple.
    pub universe_map: SmallVec<[ty::UniverseIndex; 4]>,

    /// This is equivalent to `CanonicalVarValues`, but using a
    /// `SmallVec` yields a significant performance win.
    pub var_values: SmallVec<[GenericArg<'tcx>; 8]>,
}

impl<'tcx> Default for OriginalQueryValues<'tcx> {
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
#[derive(TypeFoldable, TypeVisitable)]
pub struct CanonicalVarInfo<'tcx> {
    pub kind: CanonicalVarKind<'tcx>,
}

impl<'tcx> CanonicalVarInfo<'tcx> {
    pub fn universe(&self) -> ty::UniverseIndex {
        self.kind.universe()
    }

    #[must_use]
    pub fn with_updated_universe(self, ui: ty::UniverseIndex) -> CanonicalVarInfo<'tcx> {
        CanonicalVarInfo { kind: self.kind.with_updated_universe(ui) }
    }

    pub fn is_existential(&self) -> bool {
        match self.kind {
            CanonicalVarKind::Ty(_) => true,
            CanonicalVarKind::PlaceholderTy(_) => false,
            CanonicalVarKind::Region(_) => true,
            CanonicalVarKind::PlaceholderRegion(..) => false,
            CanonicalVarKind::Const(..) => true,
            CanonicalVarKind::PlaceholderConst(_, _) => false,
        }
    }

    pub fn is_region(&self) -> bool {
        match self.kind {
            CanonicalVarKind::Region(_) | CanonicalVarKind::PlaceholderRegion(_) => true,
            CanonicalVarKind::Ty(_)
            | CanonicalVarKind::PlaceholderTy(_)
            | CanonicalVarKind::Const(_, _)
            | CanonicalVarKind::PlaceholderConst(_, _) => false,
        }
    }

    pub fn expect_placeholder_index(self) -> usize {
        match self.kind {
            CanonicalVarKind::Ty(_)
            | CanonicalVarKind::Region(_)
            | CanonicalVarKind::Const(_, _) => bug!("expected placeholder: {self:?}"),

            CanonicalVarKind::PlaceholderRegion(placeholder) => placeholder.bound.var.as_usize(),
            CanonicalVarKind::PlaceholderTy(placeholder) => placeholder.bound.var.as_usize(),
            CanonicalVarKind::PlaceholderConst(placeholder, _) => placeholder.bound.as_usize(),
        }
    }
}

/// Describes the "kind" of the canonical variable. This is a "kind"
/// in the type-theory sense of the term -- i.e., a "meta" type system
/// that analyzes type-like values.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, TyDecodable, TyEncodable, HashStable)]
#[derive(TypeFoldable, TypeVisitable)]
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
    Const(ty::UniverseIndex, Ty<'tcx>),

    /// A "placeholder" that represents "any const".
    PlaceholderConst(ty::PlaceholderConst<'tcx>, Ty<'tcx>),
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
            CanonicalVarKind::Const(ui, _) => ui,
            CanonicalVarKind::PlaceholderConst(placeholder, _) => placeholder.universe,
        }
    }

    /// Replaces the universe of this canonical variable with `ui`.
    ///
    /// In case this is a float or int variable, this causes an ICE if
    /// the updated universe is not the root.
    pub fn with_updated_universe(self, ui: ty::UniverseIndex) -> CanonicalVarKind<'tcx> {
        match self {
            CanonicalVarKind::Ty(kind) => match kind {
                CanonicalTyVarKind::General(_) => {
                    CanonicalVarKind::Ty(CanonicalTyVarKind::General(ui))
                }
                CanonicalTyVarKind::Int | CanonicalTyVarKind::Float => {
                    assert_eq!(ui, ty::UniverseIndex::ROOT);
                    CanonicalVarKind::Ty(kind)
                }
            },
            CanonicalVarKind::PlaceholderTy(placeholder) => {
                CanonicalVarKind::PlaceholderTy(ty::Placeholder { universe: ui, ..placeholder })
            }
            CanonicalVarKind::Region(_) => CanonicalVarKind::Region(ui),
            CanonicalVarKind::PlaceholderRegion(placeholder) => {
                CanonicalVarKind::PlaceholderRegion(ty::Placeholder { universe: ui, ..placeholder })
            }
            CanonicalVarKind::Const(_, ty) => CanonicalVarKind::Const(ui, ty),
            CanonicalVarKind::PlaceholderConst(placeholder, ty) => {
                CanonicalVarKind::PlaceholderConst(
                    ty::Placeholder { universe: ui, ..placeholder },
                    ty,
                )
            }
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
#[derive(Clone, Debug, HashStable, TypeFoldable, TypeVisitable, Lift)]
pub struct QueryResponse<'tcx, R> {
    pub var_values: CanonicalVarValues<'tcx>,
    pub region_constraints: QueryRegionConstraints<'tcx>,
    pub certainty: Certainty,
    /// List of opaque types which we tried to compare to another type.
    /// Inside the query we don't know yet whether the opaque type actually
    /// should get its hidden type inferred. So we bubble the opaque type
    /// and the type it was compared against upwards and let the query caller
    /// handle it.
    pub opaque_types: Vec<(ty::OpaqueTypeKey<'tcx>, Ty<'tcx>)>,
    pub value: R,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
#[derive(HashStable, TypeFoldable, TypeVisitable, Lift)]
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

pub type CanonicalQueryResponse<'tcx, T> = &'tcx Canonical<'tcx, QueryResponse<'tcx, T>>;

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
    /// After some unification and things have been done, it makes
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
}

impl<'tcx, R> QueryResponse<'tcx, R> {
    pub fn is_proven(&self) -> bool {
        self.certainty.is_proven()
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

    /// Allows you to map the `value` of a canonical while keeping the same set of
    /// bound variables.
    ///
    /// **WARNING:** This function is very easy to mis-use, hence the name! See
    /// the comment of [Canonical::unchecked_map] for more details.
    pub fn unchecked_rebind<W>(self, value: W) -> Canonical<'tcx, W> {
        let Canonical { max_universe, variables, value: _ } = self;
        Canonical { max_universe, variables, value }
    }
}

pub type QueryOutlivesConstraint<'tcx> =
    (ty::OutlivesPredicate<GenericArg<'tcx>, Region<'tcx>>, ConstraintCategory<'tcx>);

TrivialTypeTraversalAndLiftImpls! {
    crate::infer::canonical::Certainty,
    crate::infer::canonical::CanonicalTyVarKind,
}

impl<'tcx> CanonicalVarValues<'tcx> {
    // Given a list of canonical variables, construct a set of values which are
    // the identity response.
    pub fn make_identity(
        tcx: TyCtxt<'tcx>,
        infos: CanonicalVarInfos<'tcx>,
    ) -> CanonicalVarValues<'tcx> {
        CanonicalVarValues {
            var_values: tcx.mk_substs_from_iter(infos.iter().enumerate().map(
                |(i, info)| -> ty::GenericArg<'tcx> {
                    match info.kind {
                        CanonicalVarKind::Ty(_) | CanonicalVarKind::PlaceholderTy(_) => {
                            tcx.mk_bound(ty::INNERMOST, ty::BoundVar::from_usize(i).into()).into()
                        }
                        CanonicalVarKind::Region(_) | CanonicalVarKind::PlaceholderRegion(_) => {
                            let br = ty::BoundRegion {
                                var: ty::BoundVar::from_usize(i),
                                kind: ty::BrAnon(None),
                            };
                            ty::Region::new_late_bound(tcx, ty::INNERMOST, br).into()
                        }
                        CanonicalVarKind::Const(_, ty)
                        | CanonicalVarKind::PlaceholderConst(_, ty) => ty::Const::new_bound(
                            tcx,
                            ty::INNERMOST,
                            ty::BoundVar::from_usize(i),
                            ty,
                        )
                        .into(),
                    }
                },
            )),
        }
    }

    /// Creates dummy var values which should not be used in a
    /// canonical response.
    pub fn dummy() -> CanonicalVarValues<'tcx> {
        CanonicalVarValues { var_values: ty::List::empty() }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.var_values.len()
    }
}

impl<'a, 'tcx> IntoIterator for &'a CanonicalVarValues<'tcx> {
    type Item = GenericArg<'tcx>;
    type IntoIter = ::std::iter::Copied<::std::slice::Iter<'a, GenericArg<'tcx>>>;

    fn into_iter(self) -> Self::IntoIter {
        self.var_values.iter()
    }
}

impl<'tcx> Index<BoundVar> for CanonicalVarValues<'tcx> {
    type Output = GenericArg<'tcx>;

    fn index(&self, value: BoundVar) -> &GenericArg<'tcx> {
        &self.var_values[value.as_usize()]
    }
}
