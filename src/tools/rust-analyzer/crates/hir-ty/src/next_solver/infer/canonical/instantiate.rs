//! This module contains code to instantiate new values into a
//! `Canonical<'tcx, T>`.
//!
//! For an overview of what canonicalization is and how it fits into
//! rustc, check out the [chapter in the rustc dev guide][c].
//!
//! [c]: https://rust-lang.github.io/chalk/book/canonical_queries/canonicalization.html

use crate::next_solver::{
    BoundConst, BoundRegion, BoundTy, Canonical, CanonicalVarValues, Clauses, Const, ConstKind,
    DbInterner, GenericArg, Predicate, Region, RegionKind, Ty, TyKind, fold::FnMutDelegate,
};
use rustc_hash::FxHashMap;
use rustc_type_ir::{
    BoundVarIndexKind, GenericArgKind, TypeFlags, TypeFoldable, TypeFolder, TypeSuperFoldable,
    TypeVisitableExt,
    inherent::{GenericArg as _, IntoKind, SliceLike},
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
            consts: &mut |bound_ct: BoundConst| match var_values[bound_ct.var].kind() {
                GenericArgKind::Const(ct) => ct,
                c => panic!("{bound_ct:?} is a const but value is {c:?}"),
            },
        };

        let value = tcx.replace_escaping_bound_vars_uncached(value, delegate);
        value.fold_with(&mut CanonicalInstantiator {
            tcx,
            var_values: var_values.var_values.as_slice(),
            cache: Default::default(),
        })
    }
}

/// Replaces the bound vars in a canonical binder with var values.
struct CanonicalInstantiator<'db, 'a> {
    tcx: DbInterner<'db>,

    // The values that the bound vars are being instantiated with.
    var_values: &'a [GenericArg<'db>],

    // Because we use `BoundVarIndexKind::Canonical`, we can cache
    // based only on the entire ty, not worrying about a `DebruijnIndex`
    cache: FxHashMap<Ty<'db>, Ty<'db>>,
}

impl<'db, 'a> TypeFolder<DbInterner<'db>> for CanonicalInstantiator<'db, 'a> {
    fn cx(&self) -> DbInterner<'db> {
        self.tcx
    }

    fn fold_ty(&mut self, t: Ty<'db>) -> Ty<'db> {
        match t.kind() {
            TyKind::Bound(BoundVarIndexKind::Canonical, bound_ty) => {
                self.var_values[bound_ty.var.as_usize()].expect_ty()
            }
            _ => {
                if !t.has_type_flags(TypeFlags::HAS_CANONICAL_BOUND) {
                    t
                } else if let Some(&t) = self.cache.get(&t) {
                    t
                } else {
                    let res = t.super_fold_with(self);
                    assert!(self.cache.insert(t, res).is_none());
                    res
                }
            }
        }
    }

    fn fold_region(&mut self, r: Region<'db>) -> Region<'db> {
        match r.kind() {
            RegionKind::ReBound(BoundVarIndexKind::Canonical, br) => {
                self.var_values[br.var.as_usize()].expect_region()
            }
            _ => r,
        }
    }

    fn fold_const(&mut self, ct: Const<'db>) -> Const<'db> {
        match ct.kind() {
            ConstKind::Bound(BoundVarIndexKind::Canonical, bound_const) => {
                self.var_values[bound_const.var.as_usize()].expect_const()
            }
            _ => ct.super_fold_with(self),
        }
    }

    fn fold_predicate(&mut self, p: Predicate<'db>) -> Predicate<'db> {
        if p.has_type_flags(TypeFlags::HAS_CANONICAL_BOUND) { p.super_fold_with(self) } else { p }
    }

    fn fold_clauses(&mut self, c: Clauses<'db>) -> Clauses<'db> {
        if !c.has_type_flags(TypeFlags::HAS_CANONICAL_BOUND) {
            return c;
        }

        // FIXME: We might need cache here for perf like rustc
        c.super_fold_with(self)
    }
}
