//! Inference of closure parameter types based on the closure's expected type.

use chalk_ir::{cast::Cast, AliasEq, AliasTy, FnSubst, WhereClause};
use hir_def::{expr::ExprId, HasModule};
use smallvec::SmallVec;

use crate::{
    to_chalk_trait_id, utils, ChalkTraitId, DynTy, FnPointer, FnSig, Interner, Substitution, Ty,
    TyExt, TyKind,
};

use super::{Expectation, InferenceContext};

impl InferenceContext<'_> {
    // This function handles both closures and generators.
    pub(super) fn deduce_closure_type_from_expectations(
        &mut self,
        closure_expr: ExprId,
        closure_ty: &Ty,
        sig_ty: &Ty,
        expectation: &Expectation,
    ) {
        let expected_ty = match expectation.to_option(&mut self.table) {
            Some(ty) => ty,
            None => return,
        };

        // Deduction from where-clauses in scope, as well as fn-pointer coercion are handled here.
        let _ = self.coerce(Some(closure_expr), closure_ty, &expected_ty);

        // Generators are not Fn* so return early.
        if matches!(closure_ty.kind(Interner), TyKind::Generator(..)) {
            return;
        }

        // Deduction based on the expected `dyn Fn` is done separately.
        if let TyKind::Dyn(dyn_ty) = expected_ty.kind(Interner) {
            if let Some(sig) = self.deduce_sig_from_dyn_ty(dyn_ty) {
                let expected_sig_ty = TyKind::Function(sig).intern(Interner);

                self.unify(sig_ty, &expected_sig_ty);
            }
        }
    }

    fn deduce_sig_from_dyn_ty(&self, dyn_ty: &DynTy) -> Option<FnPointer> {
        // Search for a predicate like `<$self as FnX<Args>>::Output == Ret`

        let fn_traits: SmallVec<[ChalkTraitId; 3]> =
            utils::fn_traits(self.db.upcast(), self.owner.module(self.db.upcast()).krate())
                .map(to_chalk_trait_id)
                .collect();

        let self_ty = self.result.standard_types.unknown.clone();
        let bounds = dyn_ty.bounds.clone().substitute(Interner, &[self_ty.cast(Interner)]);
        for bound in bounds.iter(Interner) {
            // NOTE(skip_binders): the extracted types are rebound by the returned `FnPointer`
            if let WhereClause::AliasEq(AliasEq { alias: AliasTy::Projection(projection), ty }) =
                bound.skip_binders()
            {
                let assoc_data = self.db.associated_ty_data(projection.associated_ty_id);
                if !fn_traits.contains(&assoc_data.trait_id) {
                    return None;
                }

                // Skip `Self`, get the type argument.
                let arg = projection.substitution.as_slice(Interner).get(1)?;
                if let Some(subst) = arg.ty(Interner)?.as_tuple() {
                    let generic_args = subst.as_slice(Interner);
                    let mut sig_tys = Vec::with_capacity(generic_args.len() + 1);
                    for arg in generic_args {
                        sig_tys.push(arg.ty(Interner)?.clone());
                    }
                    sig_tys.push(ty.clone());

                    cov_mark::hit!(dyn_fn_param_informs_call_site_closure_signature);
                    return Some(FnPointer {
                        num_binders: bound.len(Interner),
                        sig: FnSig { abi: (), safety: chalk_ir::Safety::Safe, variadic: false },
                        substitution: FnSubst(Substitution::from_iter(Interner, sig_tys)),
                    });
                }
            }
        }

        None
    }
}
