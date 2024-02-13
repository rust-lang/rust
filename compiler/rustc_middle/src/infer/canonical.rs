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

use rustc_data_structures::fx::FxHashMap;
use rustc_data_structures::sync::Lock;
use rustc_macros::HashStable;
use rustc_type_ir::Canonical as IrCanonical;
use rustc_type_ir::CanonicalVarInfo as IrCanonicalVarInfo;
pub use rustc_type_ir::{CanonicalTyVarKind, CanonicalVarKind};
use smallvec::SmallVec;
use std::collections::hash_map::Entry;
use std::ops::Index;

use crate::infer::MemberConstraint;
use crate::mir::ConstraintCategory;
use crate::ty::GenericArg;
use crate::ty::{self, BoundVar, List, Region, Ty, TyCtxt, TypeFlags, TypeVisitableExt};

pub type Canonical<'tcx, V> = IrCanonical<TyCtxt<'tcx>, V>;

pub type CanonicalVarInfo<'tcx> = IrCanonicalVarInfo<TyCtxt<'tcx>>;

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
/// `canonical_value.instantiate` to instantiate them into the canonical
/// value at the right places.
///
/// When you canonicalize a value `V`, you get back one of these
/// vectors with the original values that were replaced by canonical
/// variables. You will need to supply it later to instantiate the
/// canonicalized query response.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, TyDecodable, TyEncodable)]
#[derive(HashStable, TypeFoldable, TypeVisitable)]
pub struct CanonicalVarValues<'tcx> {
    pub var_values: ty::GenericArgsRef<'tcx>,
}

impl CanonicalVarValues<'_> {
    pub fn is_identity(&self) -> bool {
        self.var_values.iter().enumerate().all(|(bv, arg)| match arg.unpack() {
            ty::GenericArgKind::Lifetime(r) => {
                matches!(*r, ty::ReBound(ty::INNERMOST, br) if br.var.as_usize() == bv)
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
                    if let ty::ReBound(ty::INNERMOST, br) = *r
                        && var == br.var
                    {
                        var = var + 1;
                    } else {
                        // It's ok if this region var isn't unique
                    }
                }
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

/// After we execute a query with a canonicalized key, we get back a
/// `Canonical<QueryResponse<..>>`. You can use
/// `instantiate_query_result` to access the data in this result.
#[derive(Clone, Debug, HashStable, TypeFoldable, TypeVisitable)]
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
#[derive(HashStable, TypeFoldable, TypeVisitable)]
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

pub type QueryOutlivesConstraint<'tcx> =
    (ty::OutlivesPredicate<GenericArg<'tcx>, Region<'tcx>>, ConstraintCategory<'tcx>);

TrivialTypeTraversalImpls! {
    crate::infer::canonical::Certainty,
}

impl<'tcx> CanonicalVarValues<'tcx> {
    // Given a list of canonical variables, construct a set of values which are
    // the identity response.
    pub fn make_identity(
        tcx: TyCtxt<'tcx>,
        infos: CanonicalVarInfos<'tcx>,
    ) -> CanonicalVarValues<'tcx> {
        CanonicalVarValues {
            var_values: tcx.mk_args_from_iter(infos.iter().enumerate().map(
                |(i, info)| -> ty::GenericArg<'tcx> {
                    match info.kind {
                        CanonicalVarKind::Ty(_) | CanonicalVarKind::PlaceholderTy(_) => {
                            Ty::new_bound(tcx, ty::INNERMOST, ty::BoundVar::from_usize(i).into())
                                .into()
                        }
                        CanonicalVarKind::Region(_) | CanonicalVarKind::PlaceholderRegion(_) => {
                            let br = ty::BoundRegion {
                                var: ty::BoundVar::from_usize(i),
                                kind: ty::BrAnon,
                            };
                            ty::Region::new_bound(tcx, ty::INNERMOST, br).into()
                        }
                        CanonicalVarKind::Effect => ty::Const::new_bound(
                            tcx,
                            ty::INNERMOST,
                            ty::BoundVar::from_usize(i),
                            tcx.types.bool,
                        )
                        .into(),
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

#[derive(Default)]
pub struct CanonicalParamEnvCache<'tcx> {
    map: Lock<
        FxHashMap<
            ty::ParamEnv<'tcx>,
            (Canonical<'tcx, ty::ParamEnv<'tcx>>, &'tcx [GenericArg<'tcx>]),
        >,
    >,
}

impl<'tcx> CanonicalParamEnvCache<'tcx> {
    /// Gets the cached canonical form of `key` or executes
    /// `canonicalize_op` and caches the result if not present.
    ///
    /// `canonicalize_op` is intentionally not allowed to be a closure to
    /// statically prevent it from capturing `InferCtxt` and resolving
    /// inference variables, which invalidates the cache.
    pub fn get_or_insert(
        &self,
        tcx: TyCtxt<'tcx>,
        key: ty::ParamEnv<'tcx>,
        state: &mut OriginalQueryValues<'tcx>,
        canonicalize_op: fn(
            TyCtxt<'tcx>,
            ty::ParamEnv<'tcx>,
            &mut OriginalQueryValues<'tcx>,
        ) -> Canonical<'tcx, ty::ParamEnv<'tcx>>,
    ) -> Canonical<'tcx, ty::ParamEnv<'tcx>> {
        if !key.has_type_flags(
            TypeFlags::HAS_INFER | TypeFlags::HAS_PLACEHOLDER | TypeFlags::HAS_FREE_REGIONS,
        ) {
            return Canonical {
                max_universe: ty::UniverseIndex::ROOT,
                variables: List::empty(),
                value: key,
            };
        }

        assert_eq!(state.var_values.len(), 0);
        assert_eq!(state.universe_map.len(), 1);
        debug_assert_eq!(&*state.universe_map, &[ty::UniverseIndex::ROOT]);

        match self.map.borrow().entry(key) {
            Entry::Occupied(e) => {
                let (canonical, var_values) = e.get();
                state.var_values.extend_from_slice(var_values);
                *canonical
            }
            Entry::Vacant(e) => {
                let canonical = canonicalize_op(tcx, key, state);
                let OriginalQueryValues { var_values, universe_map } = state;
                assert_eq!(universe_map.len(), 1);
                e.insert((canonical, tcx.arena.alloc_slice(var_values)));
                canonical
            }
        }
    }
}
