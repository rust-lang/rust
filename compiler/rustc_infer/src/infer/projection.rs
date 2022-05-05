use rustc_middle::traits::ObligationCause;
use rustc_middle::ty::{self, ToPredicate, Ty};

use crate::traits::{Obligation, PredicateObligation};

use super::type_variable::{TypeVariableOrigin, TypeVariableOriginKind};
use super::InferCtxt;

impl<'a, 'tcx> InferCtxt<'a, 'tcx> {
    /// Instead of normalizing an associated type projection,
    /// this function generates an inference variable and registers
    /// an obligation that this inference variable must be the result
    /// of the given projection. This allows us to proceed with projections
    /// while they cannot be resolved yet due to missing information or
    /// simply due to the lack of access to the trait resolution machinery.
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
        let projection =
            ty::Binder::dummy(ty::ProjectionPredicate { projection_ty, term: ty_var.into() });
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
