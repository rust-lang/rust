//! Miscellaneous type-system utilities that are too small to deserve their own modules.

use crate::traits::{self, ObligationCause};

use rustc_data_structures::fx::FxIndexSet;
use rustc_hir as hir;
use rustc_infer::infer::outlives::env::OutlivesEnvironment;
use rustc_infer::infer::TyCtxtInferExt;
use rustc_middle::ty::{self, Ty, TyCtxt, TypeVisitable};

use crate::traits::error_reporting::TypeErrCtxtExt;

use super::outlives_bounds::InferCtxtExt;

#[derive(Clone)]
pub enum CopyImplementationError<'tcx> {
    InfrigingFields(Vec<(&'tcx ty::FieldDef, Ty<'tcx>)>),
    NotAnAdt,
    HasDestructor,
}

/// Checks that the fields of the type (an ADT) all implement copy.
///
/// If fields don't implement copy, return an error containing a list of
/// those violating fields. If it's not an ADT, returns `Err(NotAnAdt)`.
pub fn type_allowed_to_implement_copy<'tcx>(
    tcx: TyCtxt<'tcx>,
    param_env: ty::ParamEnv<'tcx>,
    self_type: Ty<'tcx>,
    parent_cause: ObligationCause<'tcx>,
) -> Result<(), CopyImplementationError<'tcx>> {
    let (adt, substs) = match self_type.kind() {
        // These types used to have a builtin impl.
        // Now libcore provides that impl.
        ty::Uint(_)
        | ty::Int(_)
        | ty::Bool
        | ty::Float(_)
        | ty::Char
        | ty::RawPtr(..)
        | ty::Never
        | ty::Ref(_, _, hir::Mutability::Not)
        | ty::Array(..) => return Ok(()),

        ty::Adt(adt, substs) => (adt, substs),

        _ => return Err(CopyImplementationError::NotAnAdt),
    };

    let copy_def_id = tcx.require_lang_item(hir::LangItem::Copy, Some(parent_cause.span));

    let mut infringing = Vec::new();
    for variant in adt.variants() {
        for field in &variant.fields {
            // Do this per-field to get better error messages.
            let infcx = tcx.infer_ctxt().build();
            let ocx = traits::ObligationCtxt::new(&infcx);

            let ty = field.ty(tcx, substs);
            if ty.references_error() {
                continue;
            }
            let span = tcx.def_span(field.did);
            // FIXME(compiler-errors): This gives us better spans for bad
            // projection types like in issue-50480.
            // If the ADT has substs, point to the cause we are given.
            // If it does not, then this field probably doesn't normalize
            // to begin with, and point to the bad field's span instead.
            let cause = if field
                .ty(tcx, traits::InternalSubsts::identity_for_item(tcx, adt.did()))
                .has_non_region_param()
            {
                parent_cause.clone()
            } else {
                ObligationCause::dummy_with_span(span)
            };

            let ty = ocx.normalize(&cause, param_env, ty);
            let normalization_errors = ocx.select_where_possible();
            if !normalization_errors.is_empty() {
                // Don't report this as a field that doesn't implement Copy,
                // but instead just implement this as a field that isn't WF.
                infcx.err_ctxt().report_fulfillment_errors(&normalization_errors, None);
                continue;
            }

            ocx.register_bound(cause, param_env, ty, copy_def_id);
            if !ocx.select_all_or_error().is_empty() {
                infringing.push((field, ty));
            }

            // Check regions assuming the self type of the impl is WF
            let outlives_env = OutlivesEnvironment::with_bounds(
                param_env,
                Some(&infcx),
                infcx.implied_bounds_tys(
                    param_env,
                    parent_cause.body_id,
                    FxIndexSet::from_iter([self_type]),
                ),
            );
            infcx.process_registered_region_obligations(
                outlives_env.region_bound_pairs(),
                param_env,
            );
            if !infcx.resolve_regions(&outlives_env).is_empty() {
                infringing.push((field, ty));
            }
        }
    }

    if !infringing.is_empty() {
        return Err(CopyImplementationError::InfrigingFields(infringing));
    }

    if adt.has_dtor(tcx) {
        return Err(CopyImplementationError::HasDestructor);
    }

    Ok(())
}
