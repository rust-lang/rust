use rustc_type_ir::data_structures::DelayedMap;
use rustc_type_ir::inherent::*;
use rustc_type_ir::{
    self as ty, InferCtxtLike, Interner, TypeFoldable, TypeFolder, TypeSuperFoldable,
    TypeVisitableExt,
};

use crate::delegate::SolverDelegate;

///////////////////////////////////////////////////////////////////////////
// EAGER RESOLUTION

/// Resolves ty, region, and const vars to their inferred values or their root vars.
pub struct EagerResolver<'a, D, I = <D as SolverDelegate>::Interner>
where
    D: SolverDelegate<Interner = I>,
    I: Interner,
{
    delegate: &'a D,
    /// We're able to use a cache here as the folder does not have any
    /// mutable state.
    cache: DelayedMap<I::Ty, I::Ty>,
}

impl<'a, D: SolverDelegate> EagerResolver<'a, D> {
    pub fn new(delegate: &'a D) -> Self {
        EagerResolver { delegate, cache: Default::default() }
    }
}

impl<D: SolverDelegate<Interner = I>, I: Interner> TypeFolder<I> for EagerResolver<'_, D> {
    fn cx(&self) -> I {
        self.delegate.cx()
    }

    fn fold_ty(&mut self, t: I::Ty) -> I::Ty {
        match t.kind() {
            ty::Infer(ty::TyVar(vid)) => {
                let resolved = self.delegate.opportunistic_resolve_ty_var(vid);
                if t != resolved && resolved.has_infer() {
                    resolved.fold_with(self)
                } else {
                    resolved
                }
            }
            ty::Infer(ty::IntVar(vid)) => self.delegate.opportunistic_resolve_int_var(vid),
            ty::Infer(ty::FloatVar(vid)) => self.delegate.opportunistic_resolve_float_var(vid),
            _ => {
                if t.has_infer() {
                    if let Some(&ty) = self.cache.get(&t) {
                        return ty;
                    }
                    let res = t.super_fold_with(self);
                    assert!(self.cache.insert(t, res));
                    res
                } else {
                    t
                }
            }
        }
    }

    fn fold_region(&mut self, r: I::Region) -> I::Region {
        match r.kind() {
            ty::ReVar(vid) => self.delegate.opportunistic_resolve_lt_var(vid),
            _ => r,
        }
    }

    fn fold_const(&mut self, c: I::Const) -> I::Const {
        match c.kind() {
            ty::ConstKind::Infer(ty::InferConst::Var(vid)) => {
                let resolved = self.delegate.opportunistic_resolve_ct_var(vid);
                if c != resolved && resolved.has_infer() {
                    resolved.fold_with(self)
                } else {
                    resolved
                }
            }
            _ => {
                if c.has_infer() {
                    c.super_fold_with(self)
                } else {
                    c
                }
            }
        }
    }

    fn fold_predicate(&mut self, p: I::Predicate) -> I::Predicate {
        if p.has_infer() { p.super_fold_with(self) } else { p }
    }
}
