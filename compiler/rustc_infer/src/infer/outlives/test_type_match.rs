use std::collections::hash_map::Entry;

use rustc_data_structures::fx::FxHashMap;
use rustc_middle::ty::TypeVisitableExt;
use rustc_middle::ty::{
    self,
    error::TypeError,
    relate::{self, Relate, RelateResult, TypeRelation},
    Ty, TyCtxt,
};

use crate::infer::region_constraints::VerifyIfEq;

/// Given a "verify-if-eq" type test like:
///
///     exists<'a...> {
///         verify_if_eq(some_type, bound_region)
///     }
///
/// and the type `test_ty` that the type test is being tested against,
/// returns:
///
/// * `None` if `some_type` cannot be made equal to `test_ty`,
///   no matter the values of the variables in `exists`.
/// * `Some(r)` with a suitable bound (typically the value of `bound_region`, modulo
///   any bound existential variables, which will be substituted) for the
///   type under test.
///
/// NB: This function uses a simplistic, syntactic version of type equality.
/// In other words, it may spuriously return `None` even if the type-under-test
/// is in fact equal to `some_type`. In practice, though, this is used on types
/// that are either projections like `T::Item` or `T` and it works fine, but it
/// could have trouble when complex types with higher-ranked binders and the
/// like are used. This is a particular challenge since this function is invoked
/// very late in inference and hence cannot make use of the normal inference
/// machinery.
#[instrument(level = "debug", skip(tcx, param_env))]
pub fn extract_verify_if_eq<'tcx>(
    tcx: TyCtxt<'tcx>,
    param_env: ty::ParamEnv<'tcx>,
    verify_if_eq_b: &ty::Binder<'tcx, VerifyIfEq<'tcx>>,
    test_ty: Ty<'tcx>,
) -> Option<ty::Region<'tcx>> {
    assert!(!verify_if_eq_b.has_escaping_bound_vars());
    let mut m = Match::new(tcx, param_env);
    let verify_if_eq = verify_if_eq_b.skip_binder();
    m.relate(verify_if_eq.ty, test_ty).ok()?;

    if let ty::RegionKind::ReLateBound(depth, br) = verify_if_eq.bound.kind() {
        assert!(depth == ty::INNERMOST);
        match m.map.get(&br) {
            Some(&r) => Some(r),
            None => {
                // If there is no mapping, then this region is unconstrained.
                // In that case, we escalate to `'static`.
                Some(tcx.lifetimes.re_static)
            }
        }
    } else {
        // The region does not contain any bound variables, so we don't need
        // to do any substitution.
        //
        // Example:
        //
        // for<'a> <T as Foo<'a>>::Item: 'b
        //
        // In this case, we've now matched and found a value for
        // `'a`, but it doesn't affect the bound `'b`.
        Some(verify_if_eq.bound)
    }
}

/// True if a (potentially higher-ranked) outlives
#[instrument(level = "debug", skip(tcx, param_env))]
pub(super) fn can_match_erased_ty<'tcx>(
    tcx: TyCtxt<'tcx>,
    param_env: ty::ParamEnv<'tcx>,
    outlives_predicate: ty::Binder<'tcx, ty::TypeOutlivesPredicate<'tcx>>,
    erased_ty: Ty<'tcx>,
) -> bool {
    assert!(!outlives_predicate.has_escaping_bound_vars());
    let erased_outlives_predicate = tcx.erase_regions(outlives_predicate);
    let outlives_ty = erased_outlives_predicate.skip_binder().0;
    if outlives_ty == erased_ty {
        // pointless micro-optimization
        true
    } else {
        Match::new(tcx, param_env).relate(outlives_ty, erased_ty).is_ok()
    }
}

struct Match<'tcx> {
    tcx: TyCtxt<'tcx>,
    param_env: ty::ParamEnv<'tcx>,
    pattern_depth: ty::DebruijnIndex,
    map: FxHashMap<ty::BoundRegion, ty::Region<'tcx>>,
}

impl<'tcx> Match<'tcx> {
    fn new(tcx: TyCtxt<'tcx>, param_env: ty::ParamEnv<'tcx>) -> Match<'tcx> {
        Match { tcx, param_env, pattern_depth: ty::INNERMOST, map: FxHashMap::default() }
    }
}

impl<'tcx> Match<'tcx> {
    /// Creates the "Error" variant that signals "no match".
    fn no_match<T>(&self) -> RelateResult<'tcx, T> {
        Err(TypeError::Mismatch)
    }

    /// Binds the pattern variable `br` to `value`; returns an `Err` if the pattern
    /// is already bound to a different value.
    #[instrument(level = "debug", skip(self))]
    fn bind(
        &mut self,
        br: ty::BoundRegion,
        value: ty::Region<'tcx>,
    ) -> RelateResult<'tcx, ty::Region<'tcx>> {
        match self.map.entry(br) {
            Entry::Occupied(entry) => {
                if *entry.get() == value {
                    Ok(value)
                } else {
                    self.no_match()
                }
            }
            Entry::Vacant(entry) => {
                entry.insert(value);
                Ok(value)
            }
        }
    }
}

impl<'tcx> TypeRelation<'tcx> for Match<'tcx> {
    fn tag(&self) -> &'static str {
        "Match"
    }

    fn intercrate(&self) -> bool {
        false
    }

    fn tcx(&self) -> TyCtxt<'tcx> {
        self.tcx
    }
    fn param_env(&self) -> ty::ParamEnv<'tcx> {
        self.param_env
    }
    fn a_is_expected(&self) -> bool {
        true
    } // irrelevant

    fn mark_ambiguous(&mut self) {
        bug!()
    }

    #[instrument(level = "trace", skip(self))]
    fn relate_with_variance<T: Relate<'tcx>>(
        &mut self,
        variance: ty::Variance,
        _: ty::VarianceDiagInfo<'tcx>,
        a: T,
        b: T,
    ) -> RelateResult<'tcx, T> {
        // Opaque types substs have lifetime parameters.
        // We must not check them to be equal, as we never insert anything to make them so.
        if variance != ty::Bivariant { self.relate(a, b) } else { Ok(a) }
    }

    #[instrument(skip(self), level = "debug")]
    fn regions(
        &mut self,
        pattern: ty::Region<'tcx>,
        value: ty::Region<'tcx>,
    ) -> RelateResult<'tcx, ty::Region<'tcx>> {
        debug!("self.pattern_depth = {:?}", self.pattern_depth);
        if let ty::RegionKind::ReLateBound(depth, br) = pattern.kind() && depth == self.pattern_depth {
            self.bind(br, value)
        } else if pattern == value {
            Ok(pattern)
        } else {
            self.no_match()
        }
    }

    #[instrument(skip(self), level = "debug")]
    fn tys(&mut self, pattern: Ty<'tcx>, value: Ty<'tcx>) -> RelateResult<'tcx, Ty<'tcx>> {
        // FIXME(non_lifetime_binders): What to do here?
        if matches!(pattern.kind(), ty::Error(_) | ty::Bound(..)) {
            // Unlike normal `TypeRelation` rules, `ty::Error` does not equal any type.
            self.no_match()
        } else if pattern == value {
            Ok(pattern)
        } else {
            relate::super_relate_tys(self, pattern, value)
        }
    }

    #[instrument(skip(self), level = "debug")]
    fn consts(
        &mut self,
        pattern: ty::Const<'tcx>,
        value: ty::Const<'tcx>,
    ) -> RelateResult<'tcx, ty::Const<'tcx>> {
        debug!("{}.consts({:?}, {:?})", self.tag(), pattern, value);
        if pattern == value {
            Ok(pattern)
        } else {
            relate::super_relate_consts(self, pattern, value)
        }
    }

    fn binders<T>(
        &mut self,
        pattern: ty::Binder<'tcx, T>,
        value: ty::Binder<'tcx, T>,
    ) -> RelateResult<'tcx, ty::Binder<'tcx, T>>
    where
        T: Relate<'tcx>,
    {
        self.pattern_depth.shift_in(1);
        let result = Ok(pattern.rebind(self.relate(pattern.skip_binder(), value.skip_binder())?));
        self.pattern_depth.shift_out(1);
        result
    }
}
