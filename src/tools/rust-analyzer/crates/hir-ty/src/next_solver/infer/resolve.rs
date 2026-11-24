//! Things for resolving vars in the infer context of the next-trait-solver.

use rustc_type_ir::{
    TypeFolder, TypeSuperFoldable, TypeVisitableExt,
    inherent::{Const as _, Ty as _},
};

use crate::next_solver::{Const, DbInterner, ErrorGuaranteed, Region, Ty};

use super::InferCtxt;

///////////////////////////////////////////////////////////////////////////
// OPPORTUNISTIC VAR RESOLVER

/// The opportunistic resolver can be used at any time. It simply replaces
/// type/const variables that have been unified with the things they have
/// been unified with (similar to `shallow_resolve`, but deep). This is
/// useful for printing messages etc but also required at various
/// points for correctness.
pub struct OpportunisticVarResolver<'a, 'db> {
    infcx: &'a InferCtxt<'db>,
}

impl<'a, 'db> OpportunisticVarResolver<'a, 'db> {
    #[inline]
    pub fn new(infcx: &'a InferCtxt<'db>) -> Self {
        OpportunisticVarResolver { infcx }
    }
}

impl<'a, 'db> TypeFolder<DbInterner<'db>> for OpportunisticVarResolver<'a, 'db> {
    fn cx(&self) -> DbInterner<'db> {
        self.infcx.interner
    }

    #[inline]
    fn fold_ty(&mut self, t: Ty<'db>) -> Ty<'db> {
        if !t.has_non_region_infer() {
            t // micro-optimize -- if there is nothing in this type that this fold affects...
        } else {
            let shallow = self.infcx.shallow_resolve(t);
            shallow.super_fold_with(self)
        }
    }

    fn fold_const(&mut self, ct: Const<'db>) -> Const<'db> {
        if !ct.has_non_region_infer() {
            ct // micro-optimize -- if there is nothing in this const that this fold affects...
        } else {
            let ct = self.infcx.shallow_resolve_const(ct);
            ct.super_fold_with(self)
        }
    }
}

pub struct ReplaceInferWithError<'db> {
    interner: DbInterner<'db>,
}

impl<'db> ReplaceInferWithError<'db> {
    #[inline]
    pub fn new(interner: DbInterner<'db>) -> Self {
        Self { interner }
    }
}

impl<'db> TypeFolder<DbInterner<'db>> for ReplaceInferWithError<'db> {
    fn cx(&self) -> DbInterner<'db> {
        self.interner
    }

    fn fold_ty(&mut self, t: Ty<'db>) -> Ty<'db> {
        if !t.has_infer() {
            return t;
        }

        if t.is_infer() {
            Ty::new_error(self.interner, ErrorGuaranteed)
        } else {
            t.super_fold_with(self)
        }
    }

    fn fold_const(&mut self, c: Const<'db>) -> Const<'db> {
        if !c.has_infer() {
            return c;
        }

        if c.is_ct_infer() {
            Const::new_error(self.interner, ErrorGuaranteed)
        } else {
            c.super_fold_with(self)
        }
    }

    fn fold_region(&mut self, r: Region<'db>) -> Region<'db> {
        if r.is_var() { Region::error(self.interner) } else { r }
    }
}
