use rustc_type_ir::fold::{TypeFoldable, TypeFolder, TypeSuperFoldable};
use rustc_type_ir::inherent::*;
use rustc_type_ir::visit::TypeVisitableExt;
use rustc_type_ir::{self as ty, InferCtxtLike, Interner};

///////////////////////////////////////////////////////////////////////////
// EAGER RESOLUTION

/// Resolves ty, region, and const vars to their inferred values or their root vars.
pub struct EagerResolver<'a, Infcx, I = <Infcx as InferCtxtLike>::Interner>
where
    Infcx: InferCtxtLike<Interner = I>,
    I: Interner,
{
    infcx: &'a Infcx,
}

impl<'a, Infcx: InferCtxtLike> EagerResolver<'a, Infcx> {
    pub fn new(infcx: &'a Infcx) -> Self {
        EagerResolver { infcx }
    }
}

impl<Infcx: InferCtxtLike<Interner = I>, I: Interner> TypeFolder<I> for EagerResolver<'_, Infcx> {
    fn interner(&self) -> I {
        self.infcx.interner()
    }

    fn fold_ty(&mut self, t: I::Ty) -> I::Ty {
        match t.kind() {
            ty::Infer(ty::TyVar(vid)) => {
                let resolved = self.infcx.opportunistic_resolve_ty_var(vid);
                if t != resolved && resolved.has_infer() {
                    resolved.fold_with(self)
                } else {
                    resolved
                }
            }
            ty::Infer(ty::IntVar(vid)) => self.infcx.opportunistic_resolve_int_var(vid),
            ty::Infer(ty::FloatVar(vid)) => self.infcx.opportunistic_resolve_float_var(vid),
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
            ty::ReVar(vid) => self.infcx.opportunistic_resolve_lt_var(vid),
            _ => r,
        }
    }

    fn fold_const(&mut self, c: I::Const) -> I::Const {
        match c.kind() {
            ty::ConstKind::Infer(ty::InferConst::Var(vid)) => {
                let resolved = self.infcx.opportunistic_resolve_ct_var(vid);
                if c != resolved && resolved.has_infer() {
                    resolved.fold_with(self)
                } else {
                    resolved
                }
            }
            ty::ConstKind::Infer(ty::InferConst::EffectVar(vid)) => {
                self.infcx.opportunistic_resolve_effect_var(vid)
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
