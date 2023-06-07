use rustc_infer::infer::at::At;
use rustc_infer::infer::type_variable::{TypeVariableOrigin, TypeVariableOriginKind};
use rustc_infer::traits::{FulfillmentError, TraitEngine};
use rustc_middle::ty::{self, Ty};

use crate::traits::{query::evaluate_obligation::InferCtxtExt, NormalizeExt, Obligation};

pub trait StructurallyNormalizeExt<'tcx> {
    fn structurally_normalize(
        &self,
        ty: Ty<'tcx>,
        fulfill_cx: &mut dyn TraitEngine<'tcx>,
    ) -> Result<Ty<'tcx>, Vec<FulfillmentError<'tcx>>>;
}

impl<'tcx> StructurallyNormalizeExt<'tcx> for At<'_, 'tcx> {
    fn structurally_normalize(
        &self,
        mut ty: Ty<'tcx>,
        fulfill_cx: &mut dyn TraitEngine<'tcx>,
    ) -> Result<Ty<'tcx>, Vec<FulfillmentError<'tcx>>> {
        assert!(!ty.is_ty_var(), "should have resolved vars before calling");

        if self.infcx.next_trait_solver() {
            while let ty::Alias(ty::Projection, projection_ty) = *ty.kind() {
                let new_infer_ty = self.infcx.next_ty_var(TypeVariableOrigin {
                    kind: TypeVariableOriginKind::NormalizeProjectionType,
                    span: self.cause.span,
                });
                let obligation = Obligation::new(
                    self.infcx.tcx,
                    self.cause.clone(),
                    self.param_env,
                    ty::Binder::dummy(ty::ProjectionPredicate {
                        projection_ty,
                        term: new_infer_ty.into(),
                    }),
                );
                if self.infcx.predicate_may_hold(&obligation) {
                    fulfill_cx.register_predicate_obligation(self.infcx, obligation);
                    let errors = fulfill_cx.select_where_possible(self.infcx);
                    if !errors.is_empty() {
                        return Err(errors);
                    }
                    ty = self.infcx.resolve_vars_if_possible(new_infer_ty);
                } else {
                    break;
                }
            }
            Ok(ty)
        } else {
            Ok(self.normalize(ty).into_value_registering_obligations(self.infcx, fulfill_cx))
        }
    }
}
