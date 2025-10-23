//! Fold impls for the next-trait-solver.

use rustc_type_ir::{
    BoundVarIndexKind, DebruijnIndex, RegionKind, TypeFoldable, TypeFolder, TypeSuperFoldable,
    TypeVisitableExt, inherent::IntoKind,
};

use crate::next_solver::BoundConst;

use super::{
    Binder, BoundRegion, BoundTy, Const, ConstKind, DbInterner, Predicate, Region, Ty, TyKind,
};

/// A delegate used when instantiating bound vars.
///
/// Any implementation must make sure that each bound variable always
/// gets mapped to the same result. `BoundVarReplacer` caches by using
/// a `DelayedMap` which does not cache the first few types it encounters.
pub trait BoundVarReplacerDelegate<'db> {
    fn replace_region(&mut self, br: BoundRegion) -> Region<'db>;
    fn replace_ty(&mut self, bt: BoundTy) -> Ty<'db>;
    fn replace_const(&mut self, bv: BoundConst) -> Const<'db>;
}

/// A simple delegate taking 3 mutable functions. The used functions must
/// always return the same result for each bound variable, no matter how
/// frequently they are called.
pub struct FnMutDelegate<'db, 'a> {
    pub regions: &'a mut (dyn FnMut(BoundRegion) -> Region<'db> + 'a),
    pub types: &'a mut (dyn FnMut(BoundTy) -> Ty<'db> + 'a),
    pub consts: &'a mut (dyn FnMut(BoundConst) -> Const<'db> + 'a),
}

impl<'db, 'a> BoundVarReplacerDelegate<'db> for FnMutDelegate<'db, 'a> {
    fn replace_region(&mut self, br: BoundRegion) -> Region<'db> {
        (self.regions)(br)
    }
    fn replace_ty(&mut self, bt: BoundTy) -> Ty<'db> {
        (self.types)(bt)
    }
    fn replace_const(&mut self, bv: BoundConst) -> Const<'db> {
        (self.consts)(bv)
    }
}

/// Replaces the escaping bound vars (late bound regions or bound types) in a type.
pub(crate) struct BoundVarReplacer<'db, D> {
    interner: DbInterner<'db>,
    /// As with `RegionFolder`, represents the index of a binder *just outside*
    /// the ones we have visited.
    current_index: DebruijnIndex,

    delegate: D,
}

impl<'db, D: BoundVarReplacerDelegate<'db>> BoundVarReplacer<'db, D> {
    pub(crate) fn new(tcx: DbInterner<'db>, delegate: D) -> Self {
        BoundVarReplacer { interner: tcx, current_index: DebruijnIndex::ZERO, delegate }
    }
}

impl<'db, D> TypeFolder<DbInterner<'db>> for BoundVarReplacer<'db, D>
where
    D: BoundVarReplacerDelegate<'db>,
{
    fn cx(&self) -> DbInterner<'db> {
        self.interner
    }

    fn fold_binder<T: TypeFoldable<DbInterner<'db>>>(
        &mut self,
        t: Binder<'db, T>,
    ) -> Binder<'db, T> {
        self.current_index.shift_in(1);
        let t = t.super_fold_with(self);
        self.current_index.shift_out(1);
        t
    }

    fn fold_ty(&mut self, t: Ty<'db>) -> Ty<'db> {
        match t.kind() {
            TyKind::Bound(BoundVarIndexKind::Bound(debruijn), bound_ty)
                if debruijn == self.current_index =>
            {
                let ty = self.delegate.replace_ty(bound_ty);
                debug_assert!(!ty.has_vars_bound_above(DebruijnIndex::ZERO));
                rustc_type_ir::shift_vars(self.interner, ty, self.current_index.as_u32())
            }
            _ => {
                if !t.has_vars_bound_at_or_above(self.current_index) {
                    t
                } else {
                    t.super_fold_with(self)
                }
            }
        }
    }

    fn fold_region(&mut self, r: Region<'db>) -> Region<'db> {
        match r.kind() {
            RegionKind::ReBound(BoundVarIndexKind::Bound(debruijn), br)
                if debruijn == self.current_index =>
            {
                let region = self.delegate.replace_region(br);
                if let RegionKind::ReBound(BoundVarIndexKind::Bound(debruijn1), br) = region.kind()
                {
                    // If the callback returns a bound region,
                    // that region should always use the INNERMOST
                    // debruijn index. Then we adjust it to the
                    // correct depth.
                    assert_eq!(debruijn1, DebruijnIndex::ZERO);
                    Region::new_bound(self.interner, debruijn, br)
                } else {
                    region
                }
            }
            _ => r,
        }
    }

    fn fold_const(&mut self, ct: Const<'db>) -> Const<'db> {
        match ct.kind() {
            ConstKind::Bound(BoundVarIndexKind::Bound(debruijn), bound_const)
                if debruijn == self.current_index =>
            {
                let ct = self.delegate.replace_const(bound_const);
                debug_assert!(!ct.has_vars_bound_above(DebruijnIndex::ZERO));
                rustc_type_ir::shift_vars(self.interner, ct, self.current_index.as_u32())
            }
            _ => ct.super_fold_with(self),
        }
    }

    fn fold_predicate(&mut self, p: Predicate<'db>) -> Predicate<'db> {
        if p.has_vars_bound_at_or_above(self.current_index) { p.super_fold_with(self) } else { p }
    }
}

pub fn fold_tys<'db, T: TypeFoldable<DbInterner<'db>>>(
    interner: DbInterner<'db>,
    t: T,
    callback: impl FnMut(Ty<'db>) -> Ty<'db>,
) -> T {
    struct Folder<'db, F> {
        interner: DbInterner<'db>,
        callback: F,
    }
    impl<'db, F: FnMut(Ty<'db>) -> Ty<'db>> TypeFolder<DbInterner<'db>> for Folder<'db, F> {
        fn cx(&self) -> DbInterner<'db> {
            self.interner
        }

        fn fold_ty(&mut self, t: Ty<'db>) -> Ty<'db> {
            let t = t.super_fold_with(self);
            (self.callback)(t)
        }
    }

    t.fold_with(&mut Folder { interner, callback })
}
