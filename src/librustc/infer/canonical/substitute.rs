//! This module contains code to substitute new values into a
//! `Canonical<'tcx, T>`.
//!
//! For an overview of what canonicalization is and how it fits into
//! rustc, check out the [chapter in the rustc guide][c].
//!
//! [c]: https://rust-lang.github.io/rustc-guide/traits/canonicalization.html

use crate::infer::canonical::{Canonical, CanonicalVarValues};
use crate::ty::fold::TypeFoldable;
use crate::ty::subst::UnpackedKind;
use crate::ty::{self, TyCtxt};

impl<'tcx, V> Canonical<'tcx, V> {
    /// Instantiate the wrapped value, replacing each canonical value
    /// with the value given in `var_values`.
    pub fn substitute(&self, tcx: TyCtxt<'tcx>, var_values: &CanonicalVarValues<'tcx>) -> V
    where
        V: TypeFoldable<'tcx>,
    {
        self.substitute_projected(tcx, var_values, |value| value)
    }

    /// Allows one to apply a substitute to some subset of
    /// `self.value`. Invoke `projection_fn` with `self.value` to get
    /// a value V that is expressed in terms of the same canonical
    /// variables bound in `self` (usually this extracts from subset
    /// of `self`). Apply the substitution `var_values` to this value
    /// V, replacing each of the canonical variables.
    pub fn substitute_projected<T>(
        &self,
        tcx: TyCtxt<'tcx>,
        var_values: &CanonicalVarValues<'tcx>,
        projection_fn: impl FnOnce(&V) -> &T,
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
pub(super) fn substitute_value<'a, 'tcx, T>(
    tcx: TyCtxt<'tcx>,
    var_values: &CanonicalVarValues<'tcx>,
    value: &'a T,
) -> T
where
    T: TypeFoldable<'tcx>,
{
    if var_values.var_values.is_empty() {
        value.clone()
    } else {
        let fld_r = |br: ty::BoundRegion| {
            match var_values.var_values[br.assert_bound_var()].unpack() {
                UnpackedKind::Lifetime(l) => l,
                r => bug!("{:?} is a region but value is {:?}", br, r),
            }
        };

        let fld_t = |bound_ty: ty::BoundTy| {
            match var_values.var_values[bound_ty.var].unpack() {
                UnpackedKind::Type(ty) => ty,
                r => bug!("{:?} is a type but value is {:?}", bound_ty, r),
            }
        };

        let fld_c = |bound_ct: ty::BoundVar, _| {
            match var_values.var_values[bound_ct].unpack() {
                UnpackedKind::Const(ct) => ct,
                c => bug!("{:?} is a const but value is {:?}", bound_ct, c),
            }
        };

        tcx.replace_escaping_bound_vars(value, fld_r, fld_t, fld_c).0
    }
}
