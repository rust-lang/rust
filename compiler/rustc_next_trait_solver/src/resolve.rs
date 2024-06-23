use crate::delegate::SolverDelegate;
use rustc_type_ir::fold::{TypeFoldable, TypeFolder, TypeSuperFoldable};
use rustc_type_ir::inherent::*;
use rustc_type_ir::visit::TypeVisitableExt;
use rustc_type_ir::{self as ty, Interner};

///////////////////////////////////////////////////////////////////////////
// EAGER RESOLUTION

/// Resolves ty, region, and const vars to their inferred values or their root vars.
pub struct EagerResolver<'a, D, I = <D as SolverDelegate>::Interner>
where
    D: SolverDelegate<Interner = I>,
    I: Interner,
{
    delegate: &'a D,
}

impl<'a, D: SolverDelegate> EagerResolver<'a, D> {
    pub fn new(delegate: &'a D) -> Self {
        EagerResolver { delegate }
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
                    t.super_fold_with(self)
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
            ty::ConstKind::Infer(ty::InferConst::EffectVar(vid)) => {
                self.delegate.opportunistic_resolve_effect_var(vid)
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
}
