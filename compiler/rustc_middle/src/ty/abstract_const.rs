//! A subset of a mir body used for const evaluability checking.
use crate::ty::{
    self, Const, EarlyBinder, Ty, TyCtxt, TypeFoldable, TypeFolder, TypeSuperFoldable,
    TypeVisitableExt,
};
use rustc_errors::ErrorGuaranteed;

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

TrivialTypeTraversalAndLiftImpls! { NotConstEvaluatable }

pub type BoundAbstractConst<'tcx> = Result<Option<EarlyBinder<ty::Const<'tcx>>>, ErrorGuaranteed>;

impl<'tcx> TyCtxt<'tcx> {
    pub fn expand_abstract_consts<T: TypeFoldable<TyCtxt<'tcx>>>(self, ac: T) -> T {
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
                    ty::ConstKind::Unevaluated(uv) => match self.tcx.thir_abstract_const(uv.def) {
                        Err(e) => ty::Const::new_error(self.tcx, e, c.ty()),
                        Ok(Some(bac)) => {
                            let args = self.tcx.erase_regions(uv.args);
                            let bac = bac.instantiate(self.tcx, args);
                            return bac.fold_with(self);
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
