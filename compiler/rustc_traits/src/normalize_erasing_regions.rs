use rustc_infer::infer::TyCtxtInferExt;
use rustc_infer::traits::ScrubbedTraitError;
use rustc_middle::bug;
use rustc_middle::query::Providers;
use rustc_middle::traits::query::NoSolution;
use rustc_middle::ty::{
    self, PseudoCanonicalInput, TyCtxt, TypeFoldable, TypeFolder, TypeSuperFoldable,
    TypeVisitableExt, TypingMode,
};
use rustc_span::DUMMY_SP;
use rustc_trait_selection::error_reporting::InferCtxtErrorExt;
use rustc_trait_selection::error_reporting::traits::OverflowCause;
use rustc_trait_selection::traits::{self, ObligationCause, ObligationCtxt};
use tracing::debug;

pub(crate) fn provide(p: &mut Providers) {
    *p = Providers {
        try_normalize_generic_arg_after_erasing_regions: |tcx, goal| {
            debug!("try_normalize_generic_arg_after_erasing_regions(goal={:#?}", goal);

            try_normalize_after_erasing_regions(tcx, goal)
        },
        ..*p
    };
}

// FIXME(-Znext-solver): This can be simplified further to just a `deeply_normalize` call.
fn try_normalize_after_erasing_regions<'tcx, T: TypeFoldable<TyCtxt<'tcx>> + PartialEq + Copy>(
    tcx: TyCtxt<'tcx>,
    goal: PseudoCanonicalInput<'tcx, T>,
) -> Result<T, NoSolution> {
    let PseudoCanonicalInput { typing_env, value } = goal;
    let (infcx, param_env) = tcx.infer_ctxt().build_with_typing_env(typing_env);
    let ocx = ObligationCtxt::new(&infcx);
    let mut normalized =
        ocx.deeply_normalize(&ObligationCause::dummy(), param_env, value).map_err(|errors| {
            match infcx.typing_mode() {
                TypingMode::PostAnalysis => {
                    for error in errors {
                        match error {
                            ScrubbedTraitError::Cycle(pred) => {
                                infcx.err_ctxt().report_overflow_error(
                                    OverflowCause::TraitSolver(pred.first().unwrap().predicate),
                                    DUMMY_SP,
                                    false,
                                    |_| {},
                                );
                            }
                            _ => {}
                        }
                    }
                }
                _ => {}
            }

            // Otherwise, bail with `NoSolution`
            NoSolution
        })?;

    if tcx.features().generic_const_exprs() {
        normalized =
            normalized.fold_with(&mut FoldConsts { ocx: &ocx, param_env, universes: vec![] });
    }

    let resolved = infcx.resolve_vars_if_possible(normalized);
    let erased = tcx.erase_regions(resolved);

    if erased.has_non_region_infer() {
        bug!("encountered infer when normalizing {value:?} to {erased:?}");
    }

    Ok(erased)
}

struct FoldConsts<'a, 'tcx> {
    ocx: &'a ObligationCtxt<'a, 'tcx>,
    param_env: ty::ParamEnv<'tcx>,
    universes: Vec<Option<ty::UniverseIndex>>,
}

impl<'tcx> TypeFolder<TyCtxt<'tcx>> for FoldConsts<'_, 'tcx> {
    fn cx(&self) -> TyCtxt<'tcx> {
        self.ocx.infcx.tcx
    }

    fn fold_binder<T>(&mut self, binder: ty::Binder<'tcx, T>) -> ty::Binder<'tcx, T>
    where
        T: TypeFoldable<TyCtxt<'tcx>>,
    {
        self.universes.push(None);
        let binder = binder.super_fold_with(self);
        self.universes.pop();
        binder
    }

    fn fold_const(&mut self, ct: ty::Const<'tcx>) -> ty::Const<'tcx> {
        let ct = traits::with_replaced_escaping_bound_vars(
            self.ocx.infcx,
            &mut self.universes,
            ct,
            |constant| traits::evaluate_const(self.ocx.infcx, constant, self.param_env),
        );
        debug!(?ct, ?self.param_env);
        ct.super_fold_with(self)
    }
}
