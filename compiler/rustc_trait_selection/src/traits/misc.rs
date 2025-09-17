//! Miscellaneous type-system utilities that are too small to deserve their own modules.

use hir::LangItem;
use rustc_ast::Mutability;
use rustc_hir as hir;
use rustc_infer::infer::{RegionResolutionError, TyCtxtInferExt};
use rustc_middle::ty::{self, AdtDef, Ty, TyCtxt, TypeVisitableExt, TypingMode};
use rustc_span::sym;

use crate::regions::InferCtxtRegionExt;
use crate::traits::{self, FulfillmentError, Obligation, ObligationCause};

pub enum CopyImplementationError<'tcx> {
    InfringingFields(Vec<(&'tcx ty::FieldDef, Ty<'tcx>, InfringingFieldsReason<'tcx>)>),
    NotAnAdt,
    HasDestructor,
    HasUnsafeFields,
}

pub enum ConstParamTyImplementationError<'tcx> {
    UnsizedConstParamsFeatureRequired,
    InvalidInnerTyOfBuiltinTy(Vec<(Ty<'tcx>, InfringingFieldsReason<'tcx>)>),
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
///
/// If the impl is `Safe`, `self_type` must not have unsafe fields. When used to
/// generate suggestions in lints, `Safe` should be supplied so as to not
/// suggest implementing `Copy` for types with unsafe fields.
pub fn type_allowed_to_implement_copy<'tcx>(
    tcx: TyCtxt<'tcx>,
    param_env: ty::ParamEnv<'tcx>,
    self_type: Ty<'tcx>,
    parent_cause: ObligationCause<'tcx>,
    impl_safety: hir::Safety,
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

    if impl_safety.is_safe() && self_type.has_unsafe_fields() {
        return Err(CopyImplementationError::HasUnsafeFields);
    }

    Ok(())
}

/// Checks that the fields of the type (an ADT) all implement `(Unsized?)ConstParamTy`.
///
/// If fields don't implement `(Unsized?)ConstParamTy`, return an error containing a list of
/// those violating fields.
///
/// If it's not an ADT, int ty, `bool` or `char`, returns `Err(NotAnAdtOrBuiltinAllowed)`.
pub fn type_allowed_to_implement_const_param_ty<'tcx>(
    tcx: TyCtxt<'tcx>,
    param_env: ty::ParamEnv<'tcx>,
    self_type: Ty<'tcx>,
    parent_cause: ObligationCause<'tcx>,
) -> Result<(), ConstParamTyImplementationError<'tcx>> {
    let mut need_unstable_feature_bound = false;

    let inner_tys: Vec<_> = match *self_type.kind() {
        // Trivially okay as these types are all:
        // - Sized
        // - Contain no nested types
        // - Have structural equality
        ty::Uint(_) | ty::Int(_) | ty::Bool | ty::Char => return Ok(()),

        // Handle types gated under `feature(unsized_const_params)`
        // FIXME(unsized_const_params): Make `const N: [u8]` work then forbid references
        ty::Slice(inner_ty) | ty::Ref(_, inner_ty, Mutability::Not) => {
            need_unstable_feature_bound = true;
            vec![inner_ty]
        }
        ty::Str => {
            need_unstable_feature_bound = true;
            vec![Ty::new_slice(tcx, tcx.types.u8)]
        }
        ty::Array(inner_ty, _) => vec![inner_ty],

        // `str` morally acts like a newtype around `[u8]`
        ty::Tuple(inner_tys) => inner_tys.into_iter().collect(),

        ty::Adt(adt, args) if adt.is_enum() || adt.is_struct() => {
            all_fields_implement_trait(
                tcx,
                param_env,
                self_type,
                adt,
                args,
                parent_cause.clone(),
                LangItem::ConstParamTy,
            )
            .map_err(ConstParamTyImplementationError::InfrigingFields)?;

            vec![]
        }

        _ => return Err(ConstParamTyImplementationError::NotAnAdtOrBuiltinAllowed),
    };

    let mut infringing_inner_tys = vec![];
    for inner_ty in inner_tys {
        // We use an ocx per inner ty for better diagnostics
        let infcx = tcx.infer_ctxt().build(TypingMode::non_body_analysis());
        let ocx = traits::ObligationCtxt::new_with_diagnostics(&infcx);

        // Make sure impls certain types are gated with #[unstable_feature_bound(unsized_const_params)]
        if need_unstable_feature_bound {
            ocx.register_obligation(Obligation::new(
                tcx,
                parent_cause.clone(),
                param_env,
                ty::ClauseKind::UnstableFeature(sym::unsized_const_params),
            ));

            if !ocx.select_all_or_error().is_empty() {
                return Err(ConstParamTyImplementationError::UnsizedConstParamsFeatureRequired);
            }
        }

        ocx.register_bound(
            parent_cause.clone(),
            param_env,
            inner_ty,
            tcx.require_lang_item(LangItem::ConstParamTy, parent_cause.span),
        );

        let errors = ocx.select_all_or_error();
        if !errors.is_empty() {
            infringing_inner_tys.push((inner_ty, InfringingFieldsReason::Fulfill(errors)));
            continue;
        }

        // Check regions assuming the self type of the impl is WF
        let errors = infcx.resolve_regions(parent_cause.body_id, param_env, [self_type]);
        if !errors.is_empty() {
            infringing_inner_tys.push((inner_ty, InfringingFieldsReason::Regions(errors)));
            continue;
        }
    }

    if !infringing_inner_tys.is_empty() {
        return Err(ConstParamTyImplementationError::InvalidInnerTyOfBuiltinTy(
            infringing_inner_tys,
        ));
    }

    Ok(())
}

/// Check that all fields of a given `adt` implement `lang_item` trait.
pub fn all_fields_implement_trait<'tcx>(
    tcx: TyCtxt<'tcx>,
    param_env: ty::ParamEnv<'tcx>,
    self_type: Ty<'tcx>,
    adt: AdtDef<'tcx>,
    args: ty::GenericArgsRef<'tcx>,
    parent_cause: ObligationCause<'tcx>,
    lang_item: LangItem,
) -> Result<(), Vec<(&'tcx ty::FieldDef, Ty<'tcx>, InfringingFieldsReason<'tcx>)>> {
    let trait_def_id = tcx.require_lang_item(lang_item, parent_cause.span);

    let mut infringing = Vec::new();
    for variant in adt.variants() {
        for field in &variant.fields {
            // Do this per-field to get better error messages.
            let infcx = tcx.infer_ctxt().build(TypingMode::non_body_analysis());
            let ocx = traits::ObligationCtxt::new_with_diagnostics(&infcx);

            let unnormalized_ty = field.ty(tcx, args);
            if unnormalized_ty.references_error() {
                continue;
            }

            let field_span = tcx.def_span(field.did);
            let field_ty_span = match tcx.hir_get_if_local(field.did) {
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
                tcx.dcx().span_delayed_bug(field_span, format!("couldn't normalize struct field `{unnormalized_ty}` when checking {tr} implementation", tr = tcx.def_path_str(trait_def_id)));
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
            let errors = infcx.resolve_regions(parent_cause.body_id, param_env, [self_type]);
            if !errors.is_empty() {
                infringing.push((field, ty, InfringingFieldsReason::Regions(errors)));
            }
        }
    }

    if infringing.is_empty() { Ok(()) } else { Err(infringing) }
}
