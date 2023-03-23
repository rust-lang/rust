use rustc_middle::traits::ObligationCause;
use rustc_middle::ty::{self, Ty};

use crate::traits::{Obligation, PredicateObligation};

use super::type_variable::{TypeVariableOrigin, TypeVariableOriginKind};
use super::InferCtxt;

impl<'tcx> InferCtxt<'tcx> {
    /// Instead of normalizing an associated type projection,
    /// this function generates an inference variable and registers
    /// an obligation that this inference variable must be the result
    /// of the given projection. This allows us to proceed with projections
    /// while they cannot be resolved yet due to missing information or
    /// simply due to the lack of access to the trait resolution machinery.
    pub fn infer_projection(
        &self,
        param_env: ty::ParamEnv<'tcx>,
        projection_ty: ty::AliasTy<'tcx>,
        cause: ObligationCause<'tcx>,
        recursion_depth: usize,
        obligations: &mut Vec<PredicateObligation<'tcx>>,
    ) -> Ty<'tcx> {
        if self.tcx.trait_solver_next() {
            // FIXME(-Ztrait-solver=next): Instead of branching here,
            // completely change the normalization routine with the new solver.
            //
            // The new solver correctly handles projection equality so this hack
            // is not necessary. if re-enabled it should emit `PredicateKind::AliasRelate`
            // not `PredicateKind::Clause(Clause::Projection(..))` as in the new solver
            // `Projection` is used as `normalizes-to` which will fail for `<T as Trait>::Assoc eq ?0`.
            return projection_ty.to_ty(self.tcx);
        } else {
            let def_id = projection_ty.def_id;
            let ty_var = self.next_ty_var(TypeVariableOrigin {
                kind: TypeVariableOriginKind::NormalizeProjectionType,
                span: self.tcx.def_span(def_id),
            });
            let projection = ty::Binder::dummy(ty::PredicateKind::Clause(ty::Clause::Projection(
                ty::ProjectionPredicate { projection_ty, term: ty_var.into() },
            )));
            let obligation =
                Obligation::with_depth(self.tcx, cause, recursion_depth, param_env, projection);
            obligations.push(obligation);
            ty_var
        }
    }
}
