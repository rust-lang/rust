//! This module contains code to instantiate new values into a
//! `Canonical<'tcx, T>`.
//!
//! For an overview of what canonicalization is and how it fits into
//! rustc, check out the [chapter in the rustc dev guide][c].
//!
//! [c]: https://rust-lang.github.io/chalk/book/canonical_queries/canonicalization.html

use rustc_macros::extension;
use rustc_middle::ty::{
    self, DelayedMap, Ty, TyCtxt, TypeFoldable, TypeFolder, TypeSuperFoldable, TypeSuperVisitable,
    TypeVisitableExt, TypeVisitor,
};
use rustc_type_ir::{TypeFlags, TypeVisitable};

use crate::infer::canonical::{Canonical, CanonicalVarValues};

/// FIXME(-Znext-solver): This or public because it is shared with the
/// new trait solver implementation. We should deduplicate canonicalization.
#[extension(pub trait CanonicalExt<'tcx, V>)]
impl<'tcx, V> Canonical<'tcx, V> {
    /// Instantiate the wrapped value, replacing each canonical value
    /// with the value given in `var_values`.
    fn instantiate(&self, tcx: TyCtxt<'tcx>, var_values: &CanonicalVarValues<'tcx>) -> V
    where
        V: TypeFoldable<TyCtxt<'tcx>>,
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
        tcx: TyCtxt<'tcx>,
        var_values: &CanonicalVarValues<'tcx>,
        projection_fn: impl FnOnce(&V) -> T,
    ) -> T
    where
        T: TypeFoldable<TyCtxt<'tcx>>,
    {
        assert_eq!(self.variables.len(), var_values.len());
        let value = projection_fn(&self.value);
        instantiate_value(tcx, var_values, value)
    }
}

/// Instantiate the values from `var_values` into `value`. `var_values`
/// must be values for the set of canonical variables that appear in
/// `value`.
pub(super) fn instantiate_value<'tcx, T>(
    tcx: TyCtxt<'tcx>,
    var_values: &CanonicalVarValues<'tcx>,
    value: T,
) -> T
where
    T: TypeFoldable<TyCtxt<'tcx>>,
{
    if var_values.var_values.is_empty() {
        return value;
    }

    value.fold_with(&mut CanonicalInstantiator {
        tcx,
        var_values: var_values.var_values,
        cache: Default::default(),
    })
}

/// Replaces the bound vars in a canonical binder with var values.
struct CanonicalInstantiator<'tcx> {
    tcx: TyCtxt<'tcx>,

    // The values that the bound vars are are being instantiated with.
    var_values: ty::GenericArgsRef<'tcx>,

    // Because we use `ty::BoundVarIndexKind::Canonical`, we can cache
    // based only on the entire ty, not worrying about a `DebruijnIndex`
    cache: DelayedMap<Ty<'tcx>, Ty<'tcx>>,
}

impl<'tcx> TypeFolder<TyCtxt<'tcx>> for CanonicalInstantiator<'tcx> {
    fn cx(&self) -> TyCtxt<'tcx> {
        self.tcx
    }

    fn fold_ty(&mut self, t: Ty<'tcx>) -> Ty<'tcx> {
        match *t.kind() {
            ty::Bound(ty::BoundVarIndexKind::Canonical, bound_ty) => {
                self.var_values[bound_ty.var.as_usize()].expect_ty()
            }
            _ => {
                if !t.has_type_flags(TypeFlags::HAS_CANONICAL_BOUND) {
                    t
                } else if let Some(&t) = self.cache.get(&t) {
                    t
                } else {
                    let res = t.super_fold_with(self);
                    assert!(self.cache.insert(t, res));
                    res
                }
            }
        }
    }

    fn fold_region(&mut self, r: ty::Region<'tcx>) -> ty::Region<'tcx> {
        match r.kind() {
            ty::ReBound(ty::BoundVarIndexKind::Canonical, br) => {
                self.var_values[br.var.as_usize()].expect_region()
            }
            _ => r,
        }
    }

    fn fold_const(&mut self, ct: ty::Const<'tcx>) -> ty::Const<'tcx> {
        match ct.kind() {
            ty::ConstKind::Bound(ty::BoundVarIndexKind::Canonical, bound_const) => {
                self.var_values[bound_const.var.as_usize()].expect_const()
            }
            _ => ct.super_fold_with(self),
        }
    }

    fn fold_predicate(&mut self, p: ty::Predicate<'tcx>) -> ty::Predicate<'tcx> {
        if p.has_type_flags(TypeFlags::HAS_CANONICAL_BOUND) { p.super_fold_with(self) } else { p }
    }

    fn fold_clauses(&mut self, c: ty::Clauses<'tcx>) -> ty::Clauses<'tcx> {
        if !c.has_type_flags(TypeFlags::HAS_CANONICAL_BOUND) {
            return c;
        }

        // Our cache key is `(clauses, var_values)`, but we also don't care about
        // var values that aren't named in the clauses, since they can change without
        // affecting the output. Since `ParamEnv`s are cached first, we compute the
        // last var value that is mentioned in the clauses, and cut off the list so
        // that we have more hits in the cache.

        // We also cache the computation of "highest var named by clauses" since that
        // is both expensive (depending on the size of the clauses) and a pure function.
        let index = *self
            .tcx
            .highest_var_in_clauses_cache
            .lock()
            .entry(c)
            .or_insert_with(|| highest_var_in_clauses(c));
        let c_args = &self.var_values[..=index];

        if let Some(c) = self.tcx.clauses_cache.lock().get(&(c, c_args)) {
            c
        } else {
            let folded = c.super_fold_with(self);
            self.tcx.clauses_cache.lock().insert((c, c_args), folded);
            folded
        }
    }
}

fn highest_var_in_clauses<'tcx>(c: ty::Clauses<'tcx>) -> usize {
    struct HighestVarInClauses {
        max_var: usize,
    }
    impl<'tcx> TypeVisitor<TyCtxt<'tcx>> for HighestVarInClauses {
        fn visit_ty(&mut self, t: Ty<'tcx>) {
            if let ty::Bound(ty::BoundVarIndexKind::Canonical, bound_ty) = *t.kind() {
                self.max_var = self.max_var.max(bound_ty.var.as_usize());
            } else if t.has_type_flags(TypeFlags::HAS_CANONICAL_BOUND) {
                t.super_visit_with(self);
            }
        }
        fn visit_region(&mut self, r: ty::Region<'tcx>) {
            if let ty::ReBound(ty::BoundVarIndexKind::Canonical, bound_region) = r.kind() {
                self.max_var = self.max_var.max(bound_region.var.as_usize());
            }
        }
        fn visit_const(&mut self, ct: ty::Const<'tcx>) {
            if let ty::ConstKind::Bound(ty::BoundVarIndexKind::Canonical, bound_const) = ct.kind() {
                self.max_var = self.max_var.max(bound_const.var.as_usize());
            } else if ct.has_type_flags(TypeFlags::HAS_CANONICAL_BOUND) {
                ct.super_visit_with(self);
            }
        }
    }
    let mut visitor = HighestVarInClauses { max_var: 0 };
    c.visit_with(&mut visitor);
    visitor.max_var
}
