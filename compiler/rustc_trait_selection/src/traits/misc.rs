//! Miscellaneous type-system utilities that are too small to deserve their own modules.

use crate::traits::{self, ObligationCause, ObligationCtxt};

use hir::LangItem;
use rustc_data_structures::fx::FxIndexSet;
use rustc_hir as hir;
use rustc_infer::infer::canonical::Canonical;
use rustc_infer::infer::{RegionResolutionError, TyCtxtInferExt};
use rustc_infer::traits::query::NoSolution;
use rustc_infer::{infer::outlives::env::OutlivesEnvironment, traits::FulfillmentError};
use rustc_middle::ty::{self, AdtDef, GenericArg, List, ParamEnv, Ty, TyCtxt, TypeVisitableExt};
use rustc_span::DUMMY_SP;

use super::outlives_bounds::InferCtxtExt;

pub enum CopyImplementationError<'tcx> {
    InfringingFields(Vec<(&'tcx ty::FieldDef, Ty<'tcx>, InfringingFieldsReason<'tcx>)>),
    NotAnAdt,
    HasDestructor,
}

pub enum ConstParamTyImplementationError<'tcx> {
    InfrigingFields(Vec<(&'tcx ty::FieldDef, Ty<'tcx>, InfringingFieldsReason<'tcx>)>),
    NotAnAdtOrBuiltinAllowed,
}

pub enum InfringingFieldsReason<'tcx> {
    Fulfill(Vec<FulfillmentError<'tcx>>),
    Regions(Vec<RegionResolutionError<'tcx>>),
}

/// Checks that the fields of the type (an ADT) all implement copy.
///
/// If fields don't implement copy, return an error containing a list of
/// those violating fields.
///
/// If it's not an ADT, int ty, `bool`, float ty, `char`, raw pointer, `!`,
/// a reference or an array returns `Err(NotAnAdt)`.
pub fn type_allowed_to_implement_copy<'tcx>(
    tcx: TyCtxt<'tcx>,
    param_env: ty::ParamEnv<'tcx>,
    self_type: Ty<'tcx>,
    parent_cause: ObligationCause<'tcx>,
) -> Result<(), CopyImplementationError<'tcx>> {
    let (adt, args) = match self_type.kind() {
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

        &ty::Adt(adt, args) => (adt, args),

        _ => return Err(CopyImplementationError::NotAnAdt),
    };

    all_fields_implement_trait(
        tcx,
        param_env,
        self_type,
        adt,
        args,
        parent_cause,
        hir::LangItem::Copy,
    )
    .map_err(CopyImplementationError::InfringingFields)?;

    if adt.has_dtor(tcx) {
        return Err(CopyImplementationError::HasDestructor);
    }

    Ok(())
}

/// Checks that the fields of the type (an ADT) all implement `ConstParamTy`.
///
/// If fields don't implement `ConstParamTy`, return an error containing a list of
/// those violating fields.
///
/// If it's not an ADT, int ty, `bool` or `char`, returns `Err(NotAnAdtOrBuiltinAllowed)`.
pub fn type_allowed_to_implement_const_param_ty<'tcx>(
    tcx: TyCtxt<'tcx>,
    param_env: ty::ParamEnv<'tcx>,
    self_type: Ty<'tcx>,
    parent_cause: ObligationCause<'tcx>,
) -> Result<(), ConstParamTyImplementationError<'tcx>> {
    let (adt, args) = match self_type.kind() {
        // `core` provides these impls.
        ty::Uint(_)
        | ty::Int(_)
        | ty::Bool
        | ty::Char
        | ty::Str
        | ty::Array(..)
        | ty::Slice(_)
        | ty::Ref(.., hir::Mutability::Not)
        | ty::Tuple(_) => return Ok(()),

        &ty::Adt(adt, args) => (adt, args),

        _ => return Err(ConstParamTyImplementationError::NotAnAdtOrBuiltinAllowed),
    };

    all_fields_implement_trait(
        tcx,
        param_env,
        self_type,
        adt,
        args,
        parent_cause,
        hir::LangItem::ConstParamTy,
    )
    .map_err(ConstParamTyImplementationError::InfrigingFields)?;

    Ok(())
}

/// Check that all fields of a given `adt` implement `lang_item` trait.
pub fn all_fields_implement_trait<'tcx>(
    tcx: TyCtxt<'tcx>,
    param_env: ty::ParamEnv<'tcx>,
    self_type: Ty<'tcx>,
    adt: AdtDef<'tcx>,
    args: &'tcx List<GenericArg<'tcx>>,
    parent_cause: ObligationCause<'tcx>,
    lang_item: LangItem,
) -> Result<(), Vec<(&'tcx ty::FieldDef, Ty<'tcx>, InfringingFieldsReason<'tcx>)>> {
    let trait_def_id = tcx.require_lang_item(lang_item, Some(parent_cause.span));

    let mut infringing = Vec::new();
    for variant in adt.variants() {
        for field in &variant.fields {
            // Do this per-field to get better error messages.
            let infcx = tcx.infer_ctxt().build();
            let ocx = traits::ObligationCtxt::new(&infcx);

            let unnormalized_ty = field.ty(tcx, args);
            if unnormalized_ty.references_error() {
                continue;
            }

            let field_span = tcx.def_span(field.did);
            let field_ty_span = match tcx.hir().get_if_local(field.did) {
                Some(hir::Node::Field(field_def)) => field_def.ty.span,
                _ => field_span,
            };

            // FIXME(compiler-errors): This gives us better spans for bad
            // projection types like in issue-50480.
            // If the ADT has args, point to the cause we are given.
            // If it does not, then this field probably doesn't normalize
            // to begin with, and point to the bad field's span instead.
            let normalization_cause = if field
                .ty(tcx, traits::GenericArgs::identity_for_item(tcx, adt.did()))
                .has_non_region_param()
            {
                parent_cause.clone()
            } else {
                ObligationCause::dummy_with_span(field_ty_span)
            };
            let ty = ocx.normalize(&normalization_cause, param_env, unnormalized_ty);
            let normalization_errors = ocx.select_where_possible();

            // NOTE: The post-normalization type may also reference errors,
            // such as when we project to a missing type or we have a mismatch
            // between expected and found const-generic types. Don't report an
            // additional copy error here, since it's not typically useful.
            if !normalization_errors.is_empty() || ty.references_error() {
                tcx.sess.delay_span_bug(field_span, format!("couldn't normalize struct field `{unnormalized_ty}` when checking {tr} implementation", tr = tcx.def_path_str(trait_def_id)));
                continue;
            }

            ocx.register_bound(
                ObligationCause::dummy_with_span(field_ty_span),
                param_env,
                ty,
                trait_def_id,
            );
            let errors = ocx.select_all_or_error();
            if !errors.is_empty() {
                infringing.push((field, ty, InfringingFieldsReason::Fulfill(errors)));
            }

            // Check regions assuming the self type of the impl is WF
            let outlives_env = OutlivesEnvironment::with_bounds(
                param_env,
                infcx.implied_bounds_tys(
                    param_env,
                    parent_cause.body_id,
                    FxIndexSet::from_iter([self_type]),
                ),
            );
            let errors = infcx.resolve_regions(&outlives_env);
            if !errors.is_empty() {
                infringing.push((field, ty, InfringingFieldsReason::Regions(errors)));
            }
        }
    }

    if infringing.is_empty() { Ok(()) } else { Err(infringing) }
}

pub fn check_tys_might_be_eq<'tcx>(
    tcx: TyCtxt<'tcx>,
    canonical: Canonical<'tcx, (ParamEnv<'tcx>, Ty<'tcx>, Ty<'tcx>)>,
) -> Result<(), NoSolution> {
    let (infcx, (param_env, ty_a, ty_b), _) =
        tcx.infer_ctxt().build_with_canonical(DUMMY_SP, &canonical);
    let ocx = ObligationCtxt::new(&infcx);

    let result = ocx.eq(&ObligationCause::dummy(), param_env, ty_a, ty_b);
    // use `select_where_possible` instead of `select_all_or_error` so that
    // we don't get errors from obligations being ambiguous.
    let errors = ocx.select_where_possible();

    if errors.len() > 0 || result.is_err() { Err(NoSolution) } else { Ok(()) }
}
