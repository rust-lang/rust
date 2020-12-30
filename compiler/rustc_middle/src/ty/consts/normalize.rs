use crate::ty::{self, Ty, TyCtxt, TypeFoldable};

impl<'tcx> TyCtxt<'tcx> {
    pub fn normalize_consts<T: TypeFoldable<'tcx>>(self, value: T) -> T {
        value.fold_with(&mut ConstNormalizer::new(self))
    }
}

pub struct ConstNormalizer<'tcx> {
    tcx: TyCtxt<'tcx>
}

impl ConstNormalizer<'_> {
    pub fn new(tcx: TyCtxt<'_>) -> ConstNormalizer<'_> {
        ConstNormalizer { tcx }
    }
}

impl<'tcx> ty::TypeFolder<'tcx> for ConstNormalizer<'tcx> {
    fn tcx(&self) -> TyCtxt<'tcx> {
        self.tcx
    }

    fn fold_ty(&mut self, t: Ty<'tcx>) -> Ty<'tcx> {
        if t.flags().intersects(ty::TypeFlags::HAS_CT_PROJECTION) {
            t.super_fold_with(self)
        } else {
            t
        }
    }

    fn fold_const(&mut self, ct: &'tcx ty::Const<'tcx>) -> &'tcx ty::Const<'tcx> {
        match ct.val {
            ty::ConstKind::Unevaluated(def, substs, None) => {
                match self.tcx.mir_abstract_const_opt_const_arg(def) {
                    // FIXME(const_evaluatable_checked): Replace the arguments not used
                    // in the abstract const with dummy ones while keeping everything that is
                    // used.
                    Ok(Some(_abstr_ct)) => self.tcx.mk_const(ty::Const {
                        ty: ct.ty,
                        val: ty::ConstKind::Unevaluated(def, substs, None)
                    }),
                    Ok(None) => {
                        let dummy_substs = ty::InternalSubsts::for_item(self.tcx, def.did, |param, _| {
                            match param.kind {
                                ty::GenericParamDefKind::Lifetime => self.tcx.lifetimes.re_static.into(),
                                ty::GenericParamDefKind::Type { .. } => self.tcx.types.unit.into(),
                                ty::GenericParamDefKind::Const => self.tcx.consts.unit.into(), // TODO
                            }
                        });
                        self.tcx.mk_const(ty::Const { ty: ct.ty, val: ty::ConstKind::Unevaluated(def, dummy_substs, None) })
                    }
                    Err(_) => self.tcx.const_error(ct.ty),
                }
            }
            _ => ct.super_fold_with(self),
        }
    }
} 
