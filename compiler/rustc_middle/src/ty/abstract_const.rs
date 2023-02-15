//! A subset of a mir body used for const evaluatability checking.
use crate::ty::{
    self, ir::TypeFolder, Const, EarlyBinder, Ty, TyCtxt, TypeFoldable, TypeSuperFoldable,
};
use rustc_errors::ErrorGuaranteed;
use rustc_hir::def_id::DefId;

#[derive(Hash, Debug, Clone, Copy, Ord, PartialOrd, PartialEq, Eq)]
#[derive(TyDecodable, TyEncodable, HashStable, TypeVisitable, TypeFoldable)]
pub enum CastKind {
    /// thir::ExprKind::As
    As,
    /// thir::ExprKind::Use
    Use,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, HashStable, TyEncodable, TyDecodable)]
pub enum NotConstEvaluatable {
    Error(ErrorGuaranteed),
    MentionsInfer,
    MentionsParam,
}

impl From<ErrorGuaranteed> for NotConstEvaluatable {
    fn from(e: ErrorGuaranteed) -> NotConstEvaluatable {
        NotConstEvaluatable::Error(e)
    }
}

TrivialTypeTraversalAndLiftImpls! {
    NotConstEvaluatable,
}

pub type BoundAbstractConst<'tcx> = Result<Option<EarlyBinder<ty::Const<'tcx>>>, ErrorGuaranteed>;

impl<'tcx> TyCtxt<'tcx> {
    /// Returns a const without substs applied
    pub fn bound_abstract_const(
        self,
        uv: ty::WithOptConstParam<DefId>,
    ) -> BoundAbstractConst<'tcx> {
        let ac = if let Some((did, param_did)) = uv.as_const_arg() {
            self.thir_abstract_const_of_const_arg((did, param_did))
        } else {
            self.thir_abstract_const(uv.did)
        };
        Ok(ac?.map(|ac| EarlyBinder(ac)))
    }

    pub fn expand_abstract_consts<T: TypeFoldable<'tcx>>(self, ac: T) -> T {
        struct Expander<'tcx> {
            tcx: TyCtxt<'tcx>,
        }

        impl<'tcx> TypeFolder<TyCtxt<'tcx>> for Expander<'tcx> {
            fn interner(&self) -> TyCtxt<'tcx> {
                self.tcx
            }
            fn fold_ty(&mut self, ty: Ty<'tcx>) -> Ty<'tcx> {
                if ty.has_type_flags(ty::TypeFlags::HAS_CT_PROJECTION) {
                    ty.super_fold_with(self)
                } else {
                    ty
                }
            }
            fn fold_const(&mut self, c: Const<'tcx>) -> Const<'tcx> {
                let ct = match c.kind() {
                    ty::ConstKind::Unevaluated(uv) => match self.tcx.bound_abstract_const(uv.def) {
                        Err(e) => self.tcx.const_error_with_guaranteed(c.ty(), e),
                        Ok(Some(bac)) => {
                            let substs = self.tcx.erase_regions(uv.substs);
                            bac.subst(self.tcx, substs)
                        }
                        Ok(None) => c,
                    },
                    _ => c,
                };
                ct.super_fold_with(self)
            }
        }
        ac.fold_with(&mut Expander { tcx: self })
    }
}
