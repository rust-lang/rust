//! This module contains code to substitute new values into a
//! `Canonical<'tcx, T>`.
//!
//! For an overview of what canonicalization is and how it fits into
//! rustc, check out the [chapter in the rustc dev guide][c].
//!
//! [c]: https://rust-lang.github.io/chalk/book/canonical_queries/canonicalization.html

use crate::infer::canonical::{Canonical, CanonicalVarValues};
use rustc_middle::ty::fold::{FnMutDelegate, TypeFoldable};
use rustc_middle::ty::subst::GenericArgKind;
use rustc_middle::ty::{self, TyCtxt};

pub(super) trait CanonicalExt<'tcx, V> {
    /// Instantiate the wrapped value, replacing each canonical value
    /// with the value given in `var_values`.
    fn substitute(&self, tcx: TyCtxt<'tcx>, var_values: &CanonicalVarValues<'tcx>) -> V
    where
        V: TypeFoldable<'tcx>;

    /// Allows one to apply a substitute to some subset of
    /// `self.value`. Invoke `projection_fn` with `self.value` to get
    /// a value V that is expressed in terms of the same canonical
    /// variables bound in `self` (usually this extracts from subset
    /// of `self`). Apply the substitution `var_values` to this value
    /// V, replacing each of the canonical variables.
    fn substitute_projected<T>(
        &self,
        tcx: TyCtxt<'tcx>,
        var_values: &CanonicalVarValues<'tcx>,
        projection_fn: impl FnOnce(&V) -> T,
    ) -> T
    where
        T: TypeFoldable<'tcx>;
}

impl<'tcx, V> CanonicalExt<'tcx, V> for Canonical<'tcx, V> {
    fn substitute(&self, tcx: TyCtxt<'tcx>, var_values: &CanonicalVarValues<'tcx>) -> V
    where
        V: TypeFoldable<'tcx>,
    {
        self.substitute_projected(tcx, var_values, |value| value.clone())
    }

    fn substitute_projected<T>(
        &self,
        tcx: TyCtxt<'tcx>,
        var_values: &CanonicalVarValues<'tcx>,
        projection_fn: impl FnOnce(&V) -> T,
    ) -> T
    where
        T: TypeFoldable<'tcx>,
    {
        assert_eq!(self.variables.len(), var_values.len());
        let value = projection_fn(&self.value);
        substitute_value(tcx, var_values, value)
    }
}

/// Substitute the values from `var_values` into `value`. `var_values`
/// must be values for the set of canonical variables that appear in
/// `value`.
pub(super) fn substitute_value<'tcx, T>(
    tcx: TyCtxt<'tcx>,
    var_values: &CanonicalVarValues<'tcx>,
    value: T,
) -> T
where
    T: TypeFoldable<'tcx>,
{
    if var_values.var_values.is_empty() {
        value
    } else {
        let delegate = FnMutDelegate {
            regions: &mut |br: ty::BoundRegion| match var_values[br.var].unpack() {
                GenericArgKind::Lifetime(l) => l,
                r => bug!("{:?} is a region but value is {:?}", br, r),
            },
            types: &mut |bound_ty: ty::BoundTy| match var_values[bound_ty.var].unpack() {
                GenericArgKind::Type(ty) => ty,
                r => bug!("{:?} is a type but value is {:?}", bound_ty, r),
            },
            consts: &mut |bound_ct: ty::BoundVar, _| match var_values[bound_ct].unpack() {
                GenericArgKind::Const(ct) => ct,
                c => bug!("{:?} is a const but value is {:?}", bound_ct, c),
            },
        };

        tcx.replace_escaping_bound_vars_uncached(value, delegate)
    }
}
