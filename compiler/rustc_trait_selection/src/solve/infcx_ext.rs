use rustc_infer::infer::at::ToTrace;
use rustc_infer::infer::type_variable::{TypeVariableOrigin, TypeVariableOriginKind};
use rustc_infer::infer::{InferCtxt, InferOk, LateBoundRegionConversionTime};
use rustc_infer::traits::query::NoSolution;
use rustc_infer::traits::ObligationCause;
use rustc_middle::infer::unify_key::{ConstVariableOrigin, ConstVariableOriginKind};
use rustc_middle::ty::{self, Ty, TypeFoldable};
use rustc_span::DUMMY_SP;

use super::Goal;

/// Methods used inside of the canonical queries of the solver.
///
/// Most notably these do not care about diagnostics information.
/// If you find this while looking for methods to use outside of the
/// solver, you may look at the implementation of these method for
/// help.
pub(super) trait InferCtxtExt<'tcx> {
    fn next_ty_infer(&self) -> Ty<'tcx>;
    fn next_const_infer(&self, ty: Ty<'tcx>) -> ty::Const<'tcx>;

    fn eq<T: ToTrace<'tcx>>(
        &self,
        param_env: ty::ParamEnv<'tcx>,
        lhs: T,
        rhs: T,
    ) -> Result<Vec<Goal<'tcx, ty::Predicate<'tcx>>>, NoSolution>;

    fn instantiate_binder_with_infer<T: TypeFoldable<'tcx> + Copy>(
        &self,
        value: ty::Binder<'tcx, T>,
    ) -> T;
}

impl<'tcx> InferCtxtExt<'tcx> for InferCtxt<'tcx> {
    fn next_ty_infer(&self) -> Ty<'tcx> {
        self.next_ty_var(TypeVariableOrigin {
            kind: TypeVariableOriginKind::MiscVariable,
            span: DUMMY_SP,
        })
    }
    fn next_const_infer(&self, ty: Ty<'tcx>) -> ty::Const<'tcx> {
        self.next_const_var(
            ty,
            ConstVariableOrigin { kind: ConstVariableOriginKind::MiscVariable, span: DUMMY_SP },
        )
    }

    #[instrument(level = "debug", skip(self, param_env), ret)]
    fn eq<T: ToTrace<'tcx>>(
        &self,
        param_env: ty::ParamEnv<'tcx>,
        lhs: T,
        rhs: T,
    ) -> Result<Vec<Goal<'tcx, ty::Predicate<'tcx>>>, NoSolution> {
        self.at(&ObligationCause::dummy(), param_env)
            .define_opaque_types(false)
            .eq(lhs, rhs)
            .map(|InferOk { value: (), obligations }| {
                obligations.into_iter().map(|o| o.into()).collect()
            })
            .map_err(|e| {
                debug!(?e, "failed to equate");
                NoSolution
            })
    }

    fn instantiate_binder_with_infer<T: TypeFoldable<'tcx> + Copy>(
        &self,
        value: ty::Binder<'tcx, T>,
    ) -> T {
        self.instantiate_binder_with_fresh_vars(
            DUMMY_SP,
            LateBoundRegionConversionTime::HigherRankedType,
            value,
        )
    }
}
