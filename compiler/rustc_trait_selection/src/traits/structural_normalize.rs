use rustc_infer::infer::at::At;
use rustc_infer::traits::TraitEngine;
use rustc_macros::extension;
use rustc_middle::ty::{self, Ty};

use crate::traits::{NormalizeExt, Obligation};

#[extension(pub trait StructurallyNormalizeExt<'tcx>)]
impl<'tcx> At<'_, 'tcx> {
    fn structurally_normalize_ty<E: 'tcx>(
        &self,
        ty: Ty<'tcx>,
        fulfill_cx: &mut dyn TraitEngine<'tcx, E>,
    ) -> Result<Ty<'tcx>, Vec<E>> {
        self.structurally_normalize_term(ty.into(), fulfill_cx).map(|term| term.expect_type())
    }

    fn structurally_normalize_const<E: 'tcx>(
        &self,
        ct: ty::Const<'tcx>,
        fulfill_cx: &mut dyn TraitEngine<'tcx, E>,
    ) -> Result<ty::Const<'tcx>, Vec<E>> {
        if self.infcx.tcx.features().generic_const_exprs() {
            return Ok(super::evaluate_const(&self.infcx, ct, self.param_env));
        }

        self.structurally_normalize_term(ct.into(), fulfill_cx).map(|term| term.expect_const())
    }

    fn structurally_normalize_term<E: 'tcx>(
        &self,
        term: ty::Term<'tcx>,
        fulfill_cx: &mut dyn TraitEngine<'tcx, E>,
    ) -> Result<ty::Term<'tcx>, Vec<E>> {
        assert!(!term.is_infer(), "should have resolved vars before calling");

        if self.infcx.next_trait_solver() {
            if let None = term.to_alias_term() {
                return Ok(term);
            }

            let new_infer = self.infcx.next_term_var_of_kind(term, self.cause.span);

            // We simply emit an `alias-eq` goal here, since that will take care of
            // normalizing the LHS of the projection until it is a rigid projection
            // (or a not-yet-defined opaque in scope).
            let obligation = Obligation::new(
                self.infcx.tcx,
                self.cause.clone(),
                self.param_env,
                ty::PredicateKind::AliasRelate(term, new_infer, ty::AliasRelationDirection::Equate),
            );

            fulfill_cx.register_predicate_obligation(self.infcx, obligation);
            let errors = fulfill_cx.select_where_possible(self.infcx);
            if !errors.is_empty() {
                return Err(errors);
            }

            Ok(self.infcx.resolve_vars_if_possible(new_infer))
        } else {
            Ok(self.normalize(term).into_value_registering_obligations(self.infcx, fulfill_cx))
        }
    }
}
