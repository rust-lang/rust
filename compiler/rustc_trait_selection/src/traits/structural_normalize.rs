use rustc_infer::infer::at::At;
use rustc_infer::infer::type_variable::{TypeVariableOrigin, TypeVariableOriginKind};
use rustc_infer::traits::{FulfillmentError, TraitEngine};
use rustc_middle::ty::{self, Ty};

use crate::traits::{NormalizeExt, Obligation};

#[extension(pub trait StructurallyNormalizeExt<'tcx>)]
impl<'tcx> At<'_, 'tcx> {
    fn structurally_normalize(
        &self,
        ty: Ty<'tcx>,
        fulfill_cx: &mut dyn TraitEngine<'tcx>,
    ) -> Result<Ty<'tcx>, Vec<FulfillmentError<'tcx>>> {
        assert!(!ty.is_ty_var(), "should have resolved vars before calling");

        if self.infcx.next_trait_solver() {
            let ty::Alias(..) = *ty.kind() else {
                return Ok(ty);
            };

            let new_infer_ty = self.infcx.next_ty_var(TypeVariableOrigin {
                kind: TypeVariableOriginKind::NormalizeProjectionType,
                span: self.cause.span,
            });

            // We simply emit an `alias-eq` goal here, since that will take care of
            // normalizing the LHS of the projection until it is a rigid projection
            // (or a not-yet-defined opaque in scope).
            let obligation = Obligation::new(
                self.infcx.tcx,
                self.cause.clone(),
                self.param_env,
                ty::PredicateKind::AliasRelate(
                    ty.into(),
                    new_infer_ty.into(),
                    ty::AliasRelationDirection::Equate,
                ),
            );

            fulfill_cx.register_predicate_obligation(self.infcx, obligation);
            let errors = fulfill_cx.select_where_possible(self.infcx);
            if !errors.is_empty() {
                return Err(errors);
            }

            Ok(self.infcx.resolve_vars_if_possible(new_infer_ty))
        } else {
            Ok(self.normalize(ty).into_value_registering_obligations(self.infcx, fulfill_cx))
        }
    }
}
