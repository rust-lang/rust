use rustc_infer::infer::at::At;
use rustc_infer::traits::{TraitEngine, TraitErrors};
use rustc_macros::extension;
use rustc_middle::ty::{self, Ty, Unnormalized};
use thin_vec::ThinVec;

use crate::traits::{NormalizeExt, Obligation};

#[extension(pub trait StructurallyNormalizeExt<'tcx>)]
impl<'tcx> At<'_, 'tcx> {
    fn structurally_normalize_ty<E: 'tcx>(
        &self,
        ty: Unnormalized<'tcx, Ty<'tcx>>,
        fulfill_cx: &mut dyn TraitEngine<'tcx, E>,
    ) -> Result<Ty<'tcx>, ThinVec<E>> {
        self.structurally_normalize_term(ty.map(Into::into), fulfill_cx)
            .map(|term| term.expect_type())
    }

    fn structurally_normalize_const<E: 'tcx>(
        &self,
        ct: Unnormalized<'tcx, ty::Const<'tcx>>,
        fulfill_cx: &mut dyn TraitEngine<'tcx, E>,
    ) -> Result<ty::Const<'tcx>, ThinVec<E>> {
        if self.infcx.tcx.features().generic_const_exprs() {
            return Ok(super::evaluate_const(&self.infcx, ct.skip_normalization(), self.param_env));
        }

        self.structurally_normalize_term(ct.map(Into::into), fulfill_cx)
            .map(|term| term.expect_const())
    }

    fn structurally_normalize_term<E: 'tcx>(
        &self,
        term: Unnormalized<'tcx, ty::Term<'tcx>>,
        fulfill_cx: &mut dyn TraitEngine<'tcx, E>,
    ) -> Result<ty::Term<'tcx>, ThinVec<E>> {
        assert!(
            !term.as_ref().skip_normalization().is_infer(),
            "should have resolved vars before calling"
        );

        if self.infcx.next_trait_solver() {
            let term = term.skip_normalization();

            if !self.infcx.tcx.renormalize_rigid_aliases() && !term.is_non_rigid_alias() {
                return Ok(term);
            };

            let Some(alias) = term.to_alias_term() else {
                return Ok(term);
            };

            let new_infer = self.infcx.next_term_var_of_alias_kind(alias, self.cause.span);

            // We simply emit an `Projection` goal here, since that will take care of
            // normalizing the LHS of the projection until it is a rigid projection
            // (or a not-yet-defined opaque in scope).
            let obligation = Obligation::new(
                self.infcx.tcx,
                self.cause.clone(),
                self.param_env,
                ty::ProjectionClause { projection_term: alias, term: new_infer },
            );

            fulfill_cx.register_predicate_obligation(self.infcx, obligation);
            let errors = fulfill_cx.try_evaluate_obligations(self.infcx);
            if let TraitErrors::HasErrors(errors) = errors {
                return Err(errors);
            }

            Ok(self.infcx.deeply_resolve_ignoring_regions(new_infer))
        } else {
            Ok(self.normalize(term).into_value_registering_obligations(self.infcx, fulfill_cx))
        }
    }
}
