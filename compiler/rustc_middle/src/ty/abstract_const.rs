//! A subset of a mir body used for const evaluatability checking.
use crate::ty::{
    self, subst::SubstsRef, Const, EarlyBinder, FallibleTypeFolder, Ty, TyCtxt, TypeFoldable,
    TypeSuperFoldable, TypeVisitable,
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
    fn bound_abstract_const(self, uv: ty::WithOptConstParam<DefId>) -> BoundAbstractConst<'tcx> {
        let ac = if let Some((did, param_did)) = uv.as_const_arg() {
            self.thir_abstract_const_of_const_arg((did, param_did))
        } else {
            self.thir_abstract_const(uv.did)
        };
        Ok(ac?.map(|ac| EarlyBinder(ac)))
    }

    pub fn expand_abstract_consts<T: TypeFoldable<'tcx>>(
        self,
        ac: T,
    ) -> Result<Option<T>, ErrorGuaranteed> {
        self._expand_abstract_consts(ac, true)
    }

    pub fn expand_unevaluated_abstract_const(
        self,
        did: ty::WithOptConstParam<DefId>,
        substs: SubstsRef<'tcx>,
    ) -> Result<Option<ty::Const<'tcx>>, ErrorGuaranteed> {
        let Some(ac) = self.bound_abstract_const(did)? else {
            return Ok(None);
        };
        let substs = self.erase_regions(substs);
        let ac = ac.subst(self, substs);
        self._expand_abstract_consts(ac, false)
    }

    fn _expand_abstract_consts<T: TypeFoldable<'tcx>>(
        self,
        ac: T,
        first: bool,
    ) -> Result<Option<T>, ErrorGuaranteed> {
        struct Expander<'tcx> {
            tcx: TyCtxt<'tcx>,
            first: bool,
        }

        impl<'tcx> FallibleTypeFolder<'tcx> for Expander<'tcx> {
            type Error = Option<ErrorGuaranteed>;
            fn tcx(&self) -> TyCtxt<'tcx> {
                self.tcx
            }
            fn try_fold_ty(&mut self, ty: Ty<'tcx>) -> Result<Ty<'tcx>, Self::Error> {
                if ty.has_type_flags(ty::TypeFlags::HAS_CT_PROJECTION) {
                    ty.try_super_fold_with(self)
                } else {
                    Ok(ty)
                }
            }
            fn try_fold_const(&mut self, c: Const<'tcx>) -> Result<Const<'tcx>, Self::Error> {
                let ct = match c.kind() {
                    ty::ConstKind::Unevaluated(uv) => {
                        if let Some(bac) = self.tcx.bound_abstract_const(uv.def)? {
                            let substs = self.tcx.erase_regions(uv.substs);
                            bac.subst(self.tcx, substs)
                        } else if self.first {
                            return Err(None);
                        } else {
                            c
                        }
                    }
                    _ => c,
                };
                self.first = false;
                ct.try_super_fold_with(self)
            }
        }
        match ac.try_fold_with(&mut Expander { tcx: self, first }) {
            Ok(c) => Ok(Some(c)),
            Err(None) => Ok(None),
            Err(Some(e)) => Err(e),
        }
    }
}
