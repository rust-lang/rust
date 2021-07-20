//! Miscellaneous type-system utilities that are too small to deserve their own modules.

use crate::infer::InferCtxtExt as _;
use crate::traits::{self, ObligationCause};

use rustc_hir as hir;
use rustc_infer::infer::TyCtxtInferExt;
use rustc_middle::ty::{self, Ty, TyCtxt, TypeFoldable};

use crate::traits::error_reporting::InferCtxtExt;

#[derive(Clone)]
pub enum CopyImplementationError<'tcx> {
    InfrigingFields(Vec<&'tcx ty::FieldDef>),
    NotAnAdt,
    HasDestructor,
}

pub fn can_type_implement_copy(
    tcx: TyCtxt<'tcx>,
    param_env: ty::ParamEnv<'tcx>,
    self_type: Ty<'tcx>,
) -> Result<(), CopyImplementationError<'tcx>> {
    // FIXME: (@jroesch) float this code up
    tcx.infer_ctxt().enter(|infcx| {
        let (adt, substs) = match self_type.kind() {
            // These types used to have a builtin impl.
            // Now libcore provides that impl.
            ty::Uint(_)
            | ty::Int(_)
            | ty::Bool
            | ty::Float(_)
            | ty::Char
            | ty::RawPtr(..)
            | ty::Never
            | ty::Ref(_, _, hir::Mutability::Not)
            | ty::Array(..) => return Ok(()),

            ty::Adt(adt, substs) => (adt, substs),

            _ => return Err(CopyImplementationError::NotAnAdt),
        };

        let mut infringing = Vec::new();
        for variant in &adt.variants {
            for field in &variant.fields {
                let ty = field.ty(tcx, substs);
                if ty.references_error() {
                    continue;
                }
                let span = tcx.def_span(field.did);
                let cause = ObligationCause::dummy_with_span(span);
                let ctx = traits::FulfillmentContext::new();
                match traits::fully_normalize(&infcx, ctx, cause, param_env, ty) {
                    Ok(ty) => {
                        if !infcx.type_is_copy_modulo_regions(param_env, ty, span) {
                            infringing.push(field);
                        }
                    }
                    Err(errors) => {
                        infcx.report_fulfillment_errors(&errors, None, false);
                    }
                };
            }
        }
        if !infringing.is_empty() {
            return Err(CopyImplementationError::InfrigingFields(infringing));
        }
        if adt.has_dtor(tcx) {
            return Err(CopyImplementationError::HasDestructor);
        }

        Ok(())
    })
}
