use rustc_middle::traits::ObligationCause;
use rustc_middle::ty::{self, ToPredicate, Ty};

use crate::traits::{Obligation, PredicateObligation};

use super::type_variable::{TypeVariableOrigin, TypeVariableOriginKind};
use super::InferCtxt;

impl<'a, 'tcx> InferCtxt<'a, 'tcx> {
    pub fn infer_projection(
        &self,
        param_env: ty::ParamEnv<'tcx>,
        projection_ty: ty::ProjectionTy<'tcx>,
        cause: ObligationCause<'tcx>,
        recursion_depth: usize,
        obligations: &mut Vec<PredicateObligation<'tcx>>,
    ) -> Ty<'tcx> {
        let def_id = projection_ty.item_def_id;
        let ty_var = self.next_ty_var(TypeVariableOrigin {
            kind: TypeVariableOriginKind::NormalizeProjectionType,
            span: self.tcx.def_span(def_id),
        });
        let projection = ty::Binder::dummy(ty::ProjectionPredicate { projection_ty, ty: ty_var });
        let obligation = Obligation::with_depth(
            cause,
            recursion_depth,
            param_env,
            projection.to_predicate(self.tcx),
        );
        obligations.push(obligation);
        ty_var
    }
}
