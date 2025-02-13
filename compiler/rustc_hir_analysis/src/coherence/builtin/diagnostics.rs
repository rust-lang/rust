use std::fmt::Debug;

use rustc_infer::traits::ObligationCauseCode;
use rustc_middle::ty::{Ty, TyCtxt, TypeFoldable, TypeFolder};
use rustc_trait_selection::traits::{self, FulfillmentError};
use rustc_type_ir::fold::TypeSuperFoldable;
use tracing::instrument;

fn redact_coerce_pointee_target_pointee<'tcx, T: Debug + TypeFoldable<TyCtxt<'tcx>>>(
    tcx: TyCtxt<'tcx>,
    target: T,
    target_pointee: Ty<'tcx>,
    new_pointee: Ty<'tcx>,
) -> T {
    struct Redactor<'tcx> {
        tcx: TyCtxt<'tcx>,
        redacted: Ty<'tcx>,
        redact_into: Ty<'tcx>,
    }

    impl<'tcx> TypeFolder<TyCtxt<'tcx>> for Redactor<'tcx> {
        fn cx(&self) -> TyCtxt<'tcx> {
            self.tcx
        }
        fn fold_ty(&mut self, ty: Ty<'tcx>) -> Ty<'tcx> {
            if ty == self.redacted {
                return self.redact_into;
            }
            ty.super_fold_with(self)
        }
    }
    target.fold_with(&mut Redactor { tcx, redacted: target_pointee, redact_into: new_pointee })
}

#[instrument(level = "debug", skip(tcx, err))]
pub(super) fn redact_fulfillment_err_for_coerce_pointee<'tcx>(
    tcx: TyCtxt<'tcx>,
    mut err: FulfillmentError<'tcx>,
    target_pointee_ty: Ty<'tcx>,
    new_pointee_ty: Ty<'tcx>,
) -> FulfillmentError<'tcx> {
    use traits::FulfillmentErrorCode::*;

    err.obligation = redact_coerce_pointee_target_pointee(
        tcx,
        err.obligation,
        target_pointee_ty,
        new_pointee_ty,
    );
    err.obligation.cause.map_code(|_| ObligationCauseCode::Misc);
    err.root_obligation = redact_coerce_pointee_target_pointee(
        tcx,
        err.root_obligation,
        target_pointee_ty,
        new_pointee_ty,
    );
    err.root_obligation.cause.map_code(|_| ObligationCauseCode::Misc);
    err.code = match err.code {
        Cycle(obs) => Cycle(
            obs.into_iter()
                .map(|obg| {
                    redact_coerce_pointee_target_pointee(
                        tcx,
                        obg,
                        target_pointee_ty,
                        new_pointee_ty,
                    )
                })
                .collect(),
        ),
        Select(selection_error) => {
            use traits::SelectionError::*;
            Select(match selection_error {
                ConstArgHasWrongType { ct, ct_ty, expected_ty } => ConstArgHasWrongType {
                    ct: redact_coerce_pointee_target_pointee(
                        tcx,
                        ct,
                        target_pointee_ty,
                        new_pointee_ty,
                    ),
                    ct_ty: redact_coerce_pointee_target_pointee(
                        tcx,
                        ct_ty,
                        target_pointee_ty,
                        new_pointee_ty,
                    ),
                    expected_ty: redact_coerce_pointee_target_pointee(
                        tcx,
                        expected_ty,
                        target_pointee_ty,
                        new_pointee_ty,
                    ),
                },
                SignatureMismatch(..)
                | Unimplemented
                | TraitDynIncompatible(..)
                | NotConstEvaluatable(..)
                | Overflow(..)
                | OpaqueTypeAutoTraitLeakageUnknown(..) => selection_error,
            })
        }
        Project(mut err) => {
            err.err = redact_coerce_pointee_target_pointee(
                tcx,
                err.err,
                target_pointee_ty,
                new_pointee_ty,
            );
            Project(err)
        }
        Subtype(expected_found, type_error) => Subtype(
            redact_coerce_pointee_target_pointee(
                tcx,
                expected_found,
                target_pointee_ty,
                new_pointee_ty,
            ),
            redact_coerce_pointee_target_pointee(
                tcx,
                type_error,
                target_pointee_ty,
                new_pointee_ty,
            ),
        ),
        ConstEquate(expected_found, type_error) => ConstEquate(
            redact_coerce_pointee_target_pointee(
                tcx,
                expected_found,
                target_pointee_ty,
                new_pointee_ty,
            ),
            redact_coerce_pointee_target_pointee(
                tcx,
                type_error,
                target_pointee_ty,
                new_pointee_ty,
            ),
        ),
        err @ Ambiguity { .. } => err,
    };
    err
}
