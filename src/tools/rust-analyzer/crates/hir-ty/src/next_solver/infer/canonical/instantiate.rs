//! This module contains code to instantiate new values into a
//! `Canonical<'tcx, T>`.
//!
//! For an overview of what canonicalization is and how it fits into
//! rustc, check out the [chapter in the rustc dev guide][c].
//!
//! [c]: https://rust-lang.github.io/chalk/book/canonical_queries/canonicalization.html

use crate::next_solver::{
    AliasTy, Binder, BoundRegion, BoundTy, Canonical, CanonicalVarValues, Const, DbInterner, Goal,
    ParamEnv, Predicate, PredicateKind, Region, Ty, TyKind,
    fold::FnMutDelegate,
    infer::{
        DefineOpaqueTypes, InferCtxt, TypeTrace,
        traits::{Obligation, PredicateObligations},
    },
};
use rustc_type_ir::{
    AliasRelationDirection, AliasTyKind, BoundVar, GenericArgKind, InferTy, TypeFoldable, Upcast,
    Variance,
    inherent::{IntoKind, SliceLike},
    relate::{
        Relate, TypeRelation, VarianceDiagInfo,
        combine::{super_combine_consts, super_combine_tys},
    },
};

pub trait CanonicalExt<'db, V> {
    fn instantiate(&self, tcx: DbInterner<'db>, var_values: &CanonicalVarValues<'db>) -> V
    where
        V: TypeFoldable<DbInterner<'db>>;
    fn instantiate_projected<T>(
        &self,
        tcx: DbInterner<'db>,
        var_values: &CanonicalVarValues<'db>,
        projection_fn: impl FnOnce(&V) -> T,
    ) -> T
    where
        T: TypeFoldable<DbInterner<'db>>;
}

/// FIXME(-Znext-solver): This or public because it is shared with the
/// new trait solver implementation. We should deduplicate canonicalization.
impl<'db, V> CanonicalExt<'db, V> for Canonical<'db, V> {
    /// Instantiate the wrapped value, replacing each canonical value
    /// with the value given in `var_values`.
    fn instantiate(&self, tcx: DbInterner<'db>, var_values: &CanonicalVarValues<'db>) -> V
    where
        V: TypeFoldable<DbInterner<'db>>,
    {
        self.instantiate_projected(tcx, var_values, |value| value.clone())
    }

    /// Allows one to apply a instantiation to some subset of
    /// `self.value`. Invoke `projection_fn` with `self.value` to get
    /// a value V that is expressed in terms of the same canonical
    /// variables bound in `self` (usually this extracts from subset
    /// of `self`). Apply the instantiation `var_values` to this value
    /// V, replacing each of the canonical variables.
    fn instantiate_projected<T>(
        &self,
        tcx: DbInterner<'db>,
        var_values: &CanonicalVarValues<'db>,
        projection_fn: impl FnOnce(&V) -> T,
    ) -> T
    where
        T: TypeFoldable<DbInterner<'db>>,
    {
        assert_eq!(self.variables.len(), var_values.len());
        let value = projection_fn(&self.value);
        instantiate_value(tcx, var_values, value)
    }
}

/// Instantiate the values from `var_values` into `value`. `var_values`
/// must be values for the set of canonical variables that appear in
/// `value`.
pub(super) fn instantiate_value<'db, T>(
    tcx: DbInterner<'db>,
    var_values: &CanonicalVarValues<'db>,
    value: T,
) -> T
where
    T: TypeFoldable<DbInterner<'db>>,
{
    if var_values.var_values.is_empty() {
        value
    } else {
        let delegate = FnMutDelegate {
            regions: &mut |br: BoundRegion| match var_values[br.var].kind() {
                GenericArgKind::Lifetime(l) => l,
                r => panic!("{br:?} is a region but value is {r:?}"),
            },
            types: &mut |bound_ty: BoundTy| match var_values[bound_ty.var].kind() {
                GenericArgKind::Type(ty) => ty,
                r => panic!("{bound_ty:?} is a type but value is {r:?}"),
            },
            consts: &mut |bound_ct: BoundVar| match var_values[bound_ct].kind() {
                GenericArgKind::Const(ct) => ct,
                c => panic!("{bound_ct:?} is a const but value is {c:?}"),
            },
        };

        tcx.replace_escaping_bound_vars_uncached(value, delegate)
    }
}
