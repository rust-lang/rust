use rustc_infer::infer::InferCtxt;
use rustc_infer::traits::{ObligationCause, TraitEngine, TraitErrors};
use rustc_macros::extension;
use rustc_middle::ty::{self, ParamEnv, Ty, Unnormalized};
use thin_vec::ThinVec;

use crate::traits::{NormalizeExt, Obligation};

#[extension(pub trait StructurallyNormalizeExt<'tcx>)]
impl<'tcx> InferCtxt<'tcx> {
    fn structurally_normalize_ty<E: 'tcx>(
        &self,
        ty: Unnormalized<'tcx, Ty<'tcx>>,
        fulfill_cx: &mut dyn TraitEngine<'tcx, E>,
        param_env: ParamEnv<'tcx>,
        cause: &ObligationCause<'tcx>,
    ) -> Result<Ty<'tcx>, ThinVec<E>> {
        self.structurally_normalize_term(ty.map(Into::into), fulfill_cx, param_env, cause)
            .map(|term| term.expect_type())
    }

    fn structurally_normalize_const<E: 'tcx>(
        &self,
        ct: Unnormalized<'tcx, ty::Const<'tcx>>,
        fulfill_cx: &mut dyn TraitEngine<'tcx, E>,
        param_env: ParamEnv<'tcx>,
        cause: &ObligationCause<'tcx>,
    ) -> Result<ty::Const<'tcx>, ThinVec<E>> {
        if self.tcx.features().generic_const_exprs() {
            return Ok(super::evaluate_const(&self, ct.skip_normalization(), param_env));
        }

        self.structurally_normalize_term(ct.map(Into::into), fulfill_cx, param_env, cause)
            .map(|term| term.expect_const())
    }

    fn structurally_normalize_term<E: 'tcx>(
        &self,
        term: Unnormalized<'tcx, ty::Term<'tcx>>,
        fulfill_cx: &mut dyn TraitEngine<'tcx, E>,
        param_env: ParamEnv<'tcx>,
        cause: &ObligationCause<'tcx>,
    ) -> Result<ty::Term<'tcx>, ThinVec<E>> {
        assert!(
            !term.as_ref().skip_normalization().is_infer(),
            "should have resolved vars before calling"
        );

        if self.next_trait_solver() {
            let term = term.skip_normalization();

            if !self.tcx.renormalize_rigid_aliases() && !term.is_non_rigid_alias() {
                return Ok(term);
            };

            let Some(alias) = term.to_alias_term() else {
                return Ok(term);
            };

            let new_infer = self.next_term_var_of_alias_kind(alias, cause.span);

            // We simply emit an `Projection` goal here, since that will take care of
            // normalizing the LHS of the projection until it is a rigid projection
            // (or a not-yet-defined opaque in scope).
            let obligation = Obligation::new(
                self.tcx,
                cause.clone(),
                param_env,
                ty::ProjectionClause { projection_term: alias, term: new_infer },
            );

            fulfill_cx.register_predicate_obligation(&self, obligation);
            let errors = fulfill_cx.try_evaluate_obligations(&self);
            if let TraitErrors::HasErrors(errors) = errors {
                return Err(errors);
            }

            Ok(self.deeply_resolve_ignoring_regions(new_infer))
        } else {
            Ok(self
                .normalize(term, param_env, cause)
                .into_value_registering_obligations(&self, fulfill_cx))
        }
    }
}
