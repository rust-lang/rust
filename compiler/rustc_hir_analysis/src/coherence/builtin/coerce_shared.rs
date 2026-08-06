use rustc_errors::ErrorGuaranteed;
use rustc_hir as hir;
use rustc_hir::ItemKind;
use rustc_hir::def_id::{DefId, LocalDefId};
use rustc_hir::lang_items::LangItem;
use rustc_infer::infer::{InferCtxt, TyCtxtInferExt};
use rustc_infer::traits::Obligation;
use rustc_middle::ty::relate::solver_relating::RelateExt;
use rustc_middle::ty::{self, Ty, TyCtxt, TypeVisitableExt, TypingMode, Unnormalized};
use rustc_span::Span;
use rustc_trait_selection::error_reporting::InferCtxtErrorExt;
use rustc_trait_selection::traits::{self, ObligationCtxt};
use tracing::debug;

use super::{
    ReborrowDataField, assert_field_type_is_copy, collect_reborrow_data_fields, field_type_is_copy,
    field_type_is_reborrow, trait_impl_lifetime_params_count,
};
use crate::diagnostics;

#[derive(Clone, Copy)]
struct CoerceSharedDiagnosticContext {
    impl_span: Span,
    trait_span: Span,
    source_ty_span: Span,
    target_ty_span: Span,
    source_lifetime_span: Option<Span>,
    target_lifetime_span: Option<Span>,
}

#[derive(Clone, Copy)]
enum CoerceSharedTypeRole {
    Source,
    Target,
}

impl CoerceSharedTypeRole {
    fn as_str(self) -> &'static str {
        match self {
            CoerceSharedTypeRole::Source => "source",
            CoerceSharedTypeRole::Target => "target",
        }
    }

    fn type_span(self, diagnostic_context: CoerceSharedDiagnosticContext) -> Span {
        match self {
            CoerceSharedTypeRole::Source => diagnostic_context.source_ty_span,
            CoerceSharedTypeRole::Target => diagnostic_context.target_ty_span,
        }
    }
}

fn coerce_shared_diagnostic_context(
    tcx: TyCtxt<'_>,
    impl_did: LocalDefId,
) -> CoerceSharedDiagnosticContext {
    let item = tcx.hir_expect_item(impl_did);
    let fallback_span = tcx.def_span(impl_did);
    let mut diagnostic_context = CoerceSharedDiagnosticContext {
        impl_span: item.span,
        trait_span: fallback_span,
        source_ty_span: fallback_span,
        target_ty_span: fallback_span,
        source_lifetime_span: None,
        target_lifetime_span: None,
    };

    let ItemKind::Impl(impl_) = &item.kind else {
        return diagnostic_context;
    };
    let Some(of_trait) = impl_.of_trait else {
        return diagnostic_context;
    };

    diagnostic_context.trait_span = of_trait.trait_ref.path.span;
    diagnostic_context.source_ty_span = impl_.self_ty.span;
    diagnostic_context.source_lifetime_span = first_explicit_lifetime_span_in_ty(impl_.self_ty)
        .or_else(|| first_explicit_impl_lifetime_param_span(impl_.generics));

    if let Some(target_ty) = coerce_shared_target_ty_from_path(of_trait.trait_ref.path) {
        diagnostic_context.target_ty_span = target_ty.span;
        diagnostic_context.target_lifetime_span =
            first_explicit_lifetime_span_in_ambig_ty(target_ty);
    } else {
        diagnostic_context.target_ty_span = diagnostic_context.trait_span;
    }

    diagnostic_context
}

fn coerce_shared_target_ty_from_path<'hir>(
    path: &'hir hir::Path<'hir>,
) -> Option<&'hir hir::Ty<'hir, hir::AmbigArg>> {
    path.segments.last()?.args().args.iter().find_map(|arg| match arg {
        hir::GenericArg::Type(ty) => Some(*ty),
        hir::GenericArg::Lifetime(_) | hir::GenericArg::Const(_) | hir::GenericArg::Infer(_) => {
            None
        }
    })
}

fn first_explicit_impl_lifetime_param_span(generics: &hir::Generics<'_>) -> Option<Span> {
    generics.params.iter().find_map(|param| match param.kind {
        hir::GenericParamKind::Lifetime { kind: hir::LifetimeParamKind::Explicit } => {
            Some(param.span)
        }
        hir::GenericParamKind::Lifetime { .. }
        | hir::GenericParamKind::Type { .. }
        | hir::GenericParamKind::Const { .. } => None,
    })
}

fn first_explicit_lifetime_span(lifetime: &hir::Lifetime) -> Option<Span> {
    match lifetime.kind {
        hir::LifetimeKind::Param(_) | hir::LifetimeKind::Static
            if !lifetime.ident.span.is_dummy() =>
        {
            Some(lifetime.ident.span)
        }
        hir::LifetimeKind::Param(_)
        | hir::LifetimeKind::Static
        | hir::LifetimeKind::ImplicitObjectLifetimeDefault
        | hir::LifetimeKind::Error(_)
        | hir::LifetimeKind::Infer => None,
    }
}

fn first_explicit_lifetime_span_in_ambig_ty(ty: &hir::Ty<'_, hir::AmbigArg>) -> Option<Span> {
    first_explicit_lifetime_span_in_ty(ty.as_unambig_ty())
}

fn first_explicit_lifetime_span_in_ty(ty: &hir::Ty<'_>) -> Option<Span> {
    match ty.kind {
        hir::TyKind::Ref(lifetime, mut_ty) => first_explicit_lifetime_span(lifetime)
            .or_else(|| first_explicit_lifetime_span_in_ty(mut_ty.ty)),
        hir::TyKind::Slice(ty)
        | hir::TyKind::Array(ty, _)
        | hir::TyKind::Pat(ty, _)
        | hir::TyKind::FieldOf(ty, _)
        | hir::TyKind::View(ty, _) => first_explicit_lifetime_span_in_ty(ty),
        hir::TyKind::Ptr(mut_ty) => first_explicit_lifetime_span_in_ty(mut_ty.ty),
        hir::TyKind::Tup(tys) => tys.iter().find_map(first_explicit_lifetime_span_in_ty),
        hir::TyKind::Path(qpath) => first_explicit_lifetime_span_in_qpath(qpath),
        hir::TyKind::TraitObject(bounds, lifetime) => bounds
            .iter()
            .find_map(|bound| first_explicit_lifetime_span_in_path(bound.trait_ref.path))
            .or_else(|| first_explicit_lifetime_span(&lifetime)),
        hir::TyKind::OpaqueDef(opaque) => first_explicit_lifetime_span_in_bounds(opaque.bounds),
        hir::TyKind::TraitAscription(bounds) => first_explicit_lifetime_span_in_bounds(bounds),
        hir::TyKind::FnPtr(fn_ptr) => {
            fn_ptr.generic_params.iter().find_map(|param| match param.kind {
                hir::GenericParamKind::Lifetime { kind: hir::LifetimeParamKind::Explicit } => {
                    Some(param.span)
                }
                hir::GenericParamKind::Lifetime { .. }
                | hir::GenericParamKind::Type { .. }
                | hir::GenericParamKind::Const { .. } => None,
            })
        }
        hir::TyKind::UnsafeBinder(binder) => binder
            .generic_params
            .iter()
            .find_map(|param| match param.kind {
                hir::GenericParamKind::Lifetime { kind: hir::LifetimeParamKind::Explicit } => {
                    Some(param.span)
                }
                hir::GenericParamKind::Lifetime { .. }
                | hir::GenericParamKind::Type { .. }
                | hir::GenericParamKind::Const { .. } => None,
            })
            .or_else(|| first_explicit_lifetime_span_in_ty(binder.inner_ty)),
        hir::TyKind::InferDelegation(_)
        | hir::TyKind::Never
        | hir::TyKind::Infer(())
        | hir::TyKind::Err(_) => None,
    }
}

fn first_explicit_lifetime_span_in_bounds(bounds: hir::GenericBounds<'_>) -> Option<Span> {
    bounds.iter().find_map(|bound| match bound {
        hir::GenericBound::Trait(poly_trait_ref) => {
            first_explicit_lifetime_span_in_path(poly_trait_ref.trait_ref.path)
        }
        hir::GenericBound::Outlives(lifetime) => first_explicit_lifetime_span(lifetime),
        hir::GenericBound::Use(args, _) => args.iter().find_map(|arg| match arg {
            hir::PreciseCapturingArgKind::Lifetime(lifetime) => {
                first_explicit_lifetime_span(lifetime)
            }
            hir::PreciseCapturingArgKind::Param(_) => None,
        }),
    })
}

fn first_explicit_lifetime_span_in_qpath(qpath: hir::QPath<'_>) -> Option<Span> {
    match qpath {
        hir::QPath::Resolved(qself, path) => qself
            .and_then(first_explicit_lifetime_span_in_ty)
            .or_else(|| first_explicit_lifetime_span_in_path(path)),
        hir::QPath::TypeRelative(qself, segment) => first_explicit_lifetime_span_in_ty(qself)
            .or_else(|| first_explicit_lifetime_span_in_path_segment(segment)),
    }
}

fn first_explicit_lifetime_span_in_path(path: &hir::Path<'_>) -> Option<Span> {
    path.segments.iter().find_map(first_explicit_lifetime_span_in_path_segment)
}

fn first_explicit_lifetime_span_in_path_segment(segment: &hir::PathSegment<'_>) -> Option<Span> {
    first_explicit_lifetime_span_in_generic_args(segment.args())
}

fn first_explicit_lifetime_span_in_generic_args(args: &hir::GenericArgs<'_>) -> Option<Span> {
    args.args
        .iter()
        .find_map(|arg| match arg {
            hir::GenericArg::Lifetime(lifetime) => first_explicit_lifetime_span(lifetime),
            hir::GenericArg::Type(ty) => first_explicit_lifetime_span_in_ambig_ty(ty),
            hir::GenericArg::Const(_) | hir::GenericArg::Infer(_) => None,
        })
        .or_else(|| {
            args.constraints.iter().find_map(|constraint| {
                first_explicit_lifetime_span_in_generic_args(constraint.gen_args)
                    .or_else(|| constraint.ty().and_then(first_explicit_lifetime_span_in_ty))
            })
        })
}

pub(super) fn coerce_shared_info<'tcx>(
    tcx: TyCtxt<'tcx>,
    impl_did: LocalDefId,
) -> Result<(), ErrorGuaranteed> {
    debug!("compute_coerce_shared_info(impl_did={:?})", impl_did);
    let infcx = tcx.infer_ctxt().build(TypingMode::non_body_analysis());
    let span = tcx.def_span(impl_did);
    let diagnostic_context = coerce_shared_diagnostic_context(tcx, impl_did);
    let trait_name = "CoerceShared";

    let coerce_shared_trait = tcx.require_lang_item(LangItem::CoerceShared, span);

    let source = tcx.type_of(impl_did).instantiate_identity().skip_norm_wip();
    let trait_ref = tcx.impl_trait_ref(impl_did).instantiate_identity().skip_norm_wip();

    if trait_impl_lifetime_params_count(tcx, impl_did) != 1 {
        return Err(tcx
            .dcx()
            .emit_err(diagnostics::CoerceSharedNotSingleLifetimeParam { span, trait_name }));
    }

    assert_eq!(trait_ref.def_id, coerce_shared_trait);
    let ocx = ObligationCtxt::new_with_diagnostics(&infcx);
    let param_env = tcx.param_env(impl_did);
    let (source, target) = ocx
        .deeply_normalize(
            &traits::ObligationCause::misc(span, impl_did),
            param_env,
            Unnormalized::new_wip((source, trait_ref.args.type_at(1))),
        )
        .map_err(|errors| infcx.err_ctxt().report_fulfillment_errors(errors))?;
    let errors = ocx.evaluate_obligations_error_on_ambiguity();
    if !errors.is_empty() {
        return Err(infcx.err_ctxt().report_fulfillment_errors(errors));
    }

    assert!(!source.has_escaping_bound_vars());

    match (source.kind(), target.kind()) {
        (&ty::Adt(def_a, args_a), &ty::Adt(def_b, args_b))
            if def_a.is_struct() && def_b.is_struct() =>
        {
            let a_lifetime = single_region_arg(args_a);
            let b_lifetime = single_region_arg(args_b);

            if a_lifetime.is_none() || b_lifetime.is_none() {
                return Err(tcx.dcx().emit_err(diagnostics::CoerceSharedMulti {
                    span: diagnostic_context.trait_span,
                    trait_name,
                }));
            }

            if a_lifetime != b_lifetime {
                return Err(tcx.dcx().emit_err(diagnostics::CoerceSharedLifetimeMismatch {
                    span: diagnostic_context.trait_span,
                    source_lifetime_span: diagnostic_context.source_lifetime_span,
                    target_lifetime_span: diagnostic_context.target_lifetime_span,
                    trait_name,
                }));
            }

            validate_reborrow_field_access(
                tcx,
                impl_did,
                def_a,
                trait_name,
                diagnostic_context,
                CoerceSharedTypeRole::Source,
            )?;
            validate_reborrow_field_access(
                tcx,
                impl_did,
                def_b,
                trait_name,
                diagnostic_context,
                CoerceSharedTypeRole::Target,
            )?;

            validate_coerce_shared_fields(
                &infcx,
                impl_did,
                param_env,
                coerce_shared_trait,
                trait_name,
                span,
                diagnostic_context,
                def_a,
                args_a,
                def_b,
                args_b,
            )
        }

        _ => {
            // Note: reusing CoerceUnsizedNonStruct error as it takes trait_name as argument.
            Err(tcx.dcx().emit_err(diagnostics::CoerceUnsizedNonStruct { span, trait_name }))
        }
    }
}

#[derive(Clone, Copy)]
struct CoerceSharedFieldPair<'tcx> {
    source: ReborrowDataField<'tcx>,
    target: ReborrowDataField<'tcx>,
}

struct CoerceSharedFields<'tcx> {
    pairs: Vec<CoerceSharedFieldPair<'tcx>>,
    unpaired_sources: Vec<ReborrowDataField<'tcx>>,
}

#[derive(Clone, Copy)]
enum CoerceSharedFieldPairError<'tcx> {
    FieldStyleMismatch,
    MissingSourceField { target: ReborrowDataField<'tcx> },
}

fn single_region_arg<'tcx>(args: ty::GenericArgsRef<'tcx>) -> Option<ty::Region<'tcx>> {
    let mut lifetimes = args.iter().filter_map(|arg| arg.as_region());
    let lifetime = lifetimes.next()?;
    lifetimes.next().is_none().then_some(lifetime)
}

// This is a coherence/WF check only. It verifies that the CoerceShared impl
// describes a structurally valid field-wise relation. Runtime lowering of the
// operation is not modeled here.
fn collect_coerce_shared_field_pairs<'tcx>(
    tcx: TyCtxt<'tcx>,
    source_def: ty::AdtDef<'tcx>,
    source_args: ty::GenericArgsRef<'tcx>,
    target_def: ty::AdtDef<'tcx>,
    target_args: ty::GenericArgsRef<'tcx>,
) -> Result<CoerceSharedFields<'tcx>, CoerceSharedFieldPairError<'tcx>> {
    let source_variant = source_def.non_enum_variant();
    let target_variant = target_def.non_enum_variant();
    if source_variant.ctor_kind() != target_variant.ctor_kind() {
        return Err(CoerceSharedFieldPairError::FieldStyleMismatch);
    }

    let source_fields = collect_reborrow_data_fields(tcx, source_def, source_args);
    let target_fields = collect_reborrow_data_fields(tcx, target_def, target_args);

    let mut pairs = Vec::with_capacity(target_fields.len());

    for target in &target_fields {
        let source = source_fields
            .iter()
            .find(|source| tcx.hygienic_eq(target.ident, source.ident, source_variant.def_id))
            .ok_or(CoerceSharedFieldPairError::MissingSourceField { target: *target })?;

        pairs.push(CoerceSharedFieldPair { source: *source, target: *target });
    }

    let unpaired_sources = source_fields
        .into_iter()
        .filter(|source| {
            !target_fields
                .iter()
                .any(|target| tcx.hygienic_eq(target.ident, source.ident, source_variant.def_id))
        })
        .collect();

    Ok(CoerceSharedFields { pairs, unpaired_sources })
}

fn validate_reborrow_field_access(
    tcx: TyCtxt<'_>,
    impl_did: LocalDefId,
    def: ty::AdtDef<'_>,
    trait_name: &'static str,
    diagnostic_context: CoerceSharedDiagnosticContext,
    role: CoerceSharedTypeRole,
) -> Result<(), ErrorGuaranteed> {
    let module = tcx.parent_module_from_def_id(impl_did);
    let variant = def.non_enum_variant();
    if variant.field_list_has_applicable_non_exhaustive() {
        return Err(tcx.dcx().emit_err(diagnostics::CoerceSharedInaccessibleField {
            span: diagnostic_context.impl_span,
            type_span: role.type_span(diagnostic_context),
            trait_name,
            role: role.as_str(),
            type_name: tcx.item_name(def.did()),
        }));
    }

    for field in &variant.fields {
        if !field.vis.is_accessible_from(module, tcx) {
            return Err(tcx.dcx().emit_err(diagnostics::CoerceSharedInaccessibleField {
                span: diagnostic_context.impl_span,
                type_span: role.type_span(diagnostic_context),
                trait_name,
                role: role.as_str(),
                type_name: tcx.item_name(def.did()),
            }));
        }
    }

    Ok(())
}

fn validate_coerce_shared_fields<'tcx>(
    infcx: &InferCtxt<'tcx>,
    impl_did: LocalDefId,
    param_env: ty::ParamEnv<'tcx>,
    coerce_shared_trait: DefId,
    trait_name: &'static str,
    span: Span,
    diagnostic_context: CoerceSharedDiagnosticContext,
    source_def: ty::AdtDef<'tcx>,
    source_args: ty::GenericArgsRef<'tcx>,
    target_def: ty::AdtDef<'tcx>,
    target_args: ty::GenericArgsRef<'tcx>,
) -> Result<(), ErrorGuaranteed> {
    let tcx = infcx.tcx;
    let fields = match collect_coerce_shared_field_pairs(
        tcx,
        source_def,
        source_args,
        target_def,
        target_args,
    ) {
        Ok(fields) => fields,
        Err(CoerceSharedFieldPairError::FieldStyleMismatch) => {
            return Err(tcx
                .dcx()
                .emit_err(diagnostics::CoerceSharedFieldStyleMismatch { span, trait_name }));
        }
        Err(CoerceSharedFieldPairError::MissingSourceField { target }) => {
            return Err(tcx.dcx().emit_err(diagnostics::CoerceSharedMissingField {
                span: target.span,
                source_ty_span: diagnostic_context.source_ty_span,
                trait_name,
                source_ty_name: tcx.item_name(source_def.did()),
                field_name: target.name,
            }));
        }
    };

    for field_pair in fields.pairs {
        validate_coerce_shared_field(
            infcx,
            impl_did,
            param_env,
            coerce_shared_trait,
            trait_name,
            span,
            diagnostic_context,
            field_pair.source,
            field_pair.target,
        )?;
    }

    let reborrow_trait = tcx.require_lang_item(LangItem::Reborrow, span);
    for source in fields.unpaired_sources {
        validate_coerce_shared_unpaired_source_field(
            infcx,
            impl_did,
            param_env,
            reborrow_trait,
            trait_name,
            diagnostic_context,
            source,
        )?;
    }

    // FIXME(reborrow): remove this temporary WF-side memcpy-ability guard once
    // the downstream CoerceShared implementation can correctly handle source and
    // target types that are not trivially memcpy-able. Refer to #157489
    validate_coerce_shared_fields_are_memcpy_compatible(
        infcx,
        impl_did,
        param_env,
        coerce_shared_trait,
        trait_name,
        span,
        diagnostic_context,
        source_def,
        source_args,
        target_def,
        target_args,
    )?;

    Ok(())
}

fn validate_coerce_shared_fields_are_memcpy_compatible<'tcx>(
    infcx: &InferCtxt<'tcx>,
    impl_did: LocalDefId,
    param_env: ty::ParamEnv<'tcx>,
    coerce_shared_trait: DefId,
    trait_name: &'static str,
    span: Span,
    diagnostic_context: CoerceSharedDiagnosticContext,
    source_def: ty::AdtDef<'tcx>,
    source_args: ty::GenericArgsRef<'tcx>,
    target_def: ty::AdtDef<'tcx>,
    target_args: ty::GenericArgsRef<'tcx>,
) -> Result<(), ErrorGuaranteed> {
    let tcx = infcx.tcx;
    let source_non_zst_fields =
        non_zst_reborrow_data_fields(infcx, param_env, source_def, source_args);
    let target_non_zst_fields =
        non_zst_reborrow_data_fields(infcx, param_env, target_def, target_args);

    match (&source_non_zst_fields[..], &target_non_zst_fields[..]) {
        ([], []) => Ok(()),
        ([source], [target]) => {
            if field_tys_satisfy_relation_after_normalization_and_resolution(
                tcx,
                impl_did,
                param_env,
                source.ty,
                target.ty,
                source.span,
                FieldRelation::Equal,
            ) {
                return Ok(());
            }

            if matches!(
                (source.ty.kind(), target.ty.kind()),
                (&ty::Ref(_, _, ty::Mutability::Mut), &ty::Ref(_, _, ty::Mutability::Not))
                    | (&ty::Alias(..), _)
                    | (_, &ty::Alias(..))
            ) && field_tys_satisfy_relation_after_normalization_and_resolution(
                tcx,
                impl_did,
                param_env,
                source.ty,
                target.ty,
                source.span,
                FieldRelation::MutRefToSharedRef,
            ) {
                return Ok(());
            }

            validate_field_tys_satisfy_coerce_shared_relation(
                infcx,
                impl_did,
                param_env,
                coerce_shared_trait,
                trait_name,
                span,
                diagnostic_context,
                *source,
                *target,
            )
        }
        _ => Err(tcx.dcx().emit_err(diagnostics::CoerceSharedMultipleNonZstFields {
            span: diagnostic_context.impl_span,
            source_ty_span: diagnostic_context.source_ty_span,
            target_ty_span: diagnostic_context.target_ty_span,
            trait_name,
            source_count: source_non_zst_fields.len(),
            target_count: target_non_zst_fields.len(),
        })),
    }
}

fn non_zst_reborrow_data_fields<'tcx>(
    infcx: &InferCtxt<'tcx>,
    param_env: ty::ParamEnv<'tcx>,
    def: ty::AdtDef<'tcx>,
    args: ty::GenericArgsRef<'tcx>,
) -> Vec<ReborrowDataField<'tcx>> {
    let tcx = infcx.tcx;
    collect_reborrow_data_fields(tcx, def, args)
        .into_iter()
        .filter(|field| {
            !matches!(
                tcx.layout_of(infcx.typing_env(param_env).as_query_input(field.ty)),
                Ok(layout) if layout.is_zst()
            )
        })
        .collect()
}

fn validate_coerce_shared_field<'tcx>(
    infcx: &InferCtxt<'tcx>,
    impl_did: LocalDefId,
    param_env: ty::ParamEnv<'tcx>,
    coerce_shared_trait: DefId,
    trait_name: &'static str,
    span: Span,
    diagnostic_context: CoerceSharedDiagnosticContext,
    source: ReborrowDataField<'tcx>,
    target: ReborrowDataField<'tcx>,
) -> Result<(), ErrorGuaranteed> {
    let tcx = infcx.tcx;
    if matches!(
        (source.ty.kind(), target.ty.kind()),
        (&ty::Ref(_, _, ty::Mutability::Mut), &ty::Ref(_, _, ty::Mutability::Not))
            | (&ty::Alias(..), _)
            | (_, &ty::Alias(..))
    ) && field_tys_satisfy_relation_after_normalization_and_resolution(
        tcx,
        impl_did,
        param_env,
        source.ty,
        target.ty,
        source.span,
        FieldRelation::MutRefToSharedRef,
    ) {
        return Ok(());
    }

    if field_tys_satisfy_relation_after_normalization_and_resolution(
        tcx,
        impl_did,
        param_env,
        source.ty,
        target.ty,
        source.span,
        FieldRelation::Equal,
    ) {
        return assert_field_type_is_copy(tcx, infcx, impl_did, param_env, source.ty, source.span);
    }

    validate_field_tys_satisfy_coerce_shared_relation(
        infcx,
        impl_did,
        param_env,
        coerce_shared_trait,
        trait_name,
        span,
        diagnostic_context,
        source,
        target,
    )
}

fn validate_coerce_shared_unpaired_source_field<'tcx>(
    infcx: &InferCtxt<'tcx>,
    impl_did: LocalDefId,
    param_env: ty::ParamEnv<'tcx>,
    reborrow_trait: DefId,
    trait_name: &'static str,
    diagnostic_context: CoerceSharedDiagnosticContext,
    mut source: ReborrowDataField<'tcx>,
) -> Result<(), ErrorGuaranteed> {
    let tcx = infcx.tcx;
    let ocx = ObligationCtxt::new_with_diagnostics(infcx);
    source.ty = ocx
        .deeply_normalize(
            &traits::ObligationCause::misc(source.span, impl_did),
            param_env,
            Unnormalized::new_wip(source.ty),
        )
        .map_err(|errors| infcx.err_ctxt().report_fulfillment_errors(errors))?;

    if field_type_is_reborrow(
        tcx,
        infcx,
        reborrow_trait,
        impl_did,
        param_env,
        source.ty,
        source.span,
    ) || field_type_is_copy(tcx, infcx, impl_did, param_env, source.ty, source.span)
    {
        return Ok(());
    }

    Err(tcx.dcx().emit_err(diagnostics::CoerceSharedOmittedSourceFieldNotCopyOrReborrow {
        span: source.span,
        impl_span: diagnostic_context.impl_span,
        trait_name,
        field_name: source.name,
        field_ty: source.ty,
    }))
}

fn validate_field_tys_satisfy_coerce_shared_relation<'tcx>(
    infcx: &InferCtxt<'tcx>,
    impl_did: LocalDefId,
    param_env: ty::ParamEnv<'tcx>,
    coerce_shared_trait: DefId,
    trait_name: &'static str,
    span: Span,
    diagnostic_context: CoerceSharedDiagnosticContext,
    source: ReborrowDataField<'tcx>,
    target: ReborrowDataField<'tcx>,
) -> Result<(), ErrorGuaranteed> {
    let tcx = infcx.tcx;
    let ocx = ObligationCtxt::new_with_diagnostics(infcx);
    let cause = traits::ObligationCause::misc(span, impl_did);
    ocx.register_obligation(Obligation::new(
        tcx,
        cause,
        param_env,
        ty::TraitRef::new(tcx, coerce_shared_trait, [source.ty, target.ty]),
    ));
    let errors = ocx.evaluate_obligations_error_on_ambiguity();

    if !errors.is_empty() {
        return Err(emit_coerce_shared_field_mismatch(
            tcx,
            trait_name,
            diagnostic_context,
            source,
            target,
        ));
    }

    ocx.resolve_regions_and_report_errors(impl_did, param_env, [])
}

fn emit_coerce_shared_field_mismatch<'tcx>(
    tcx: TyCtxt<'tcx>,
    trait_name: &'static str,
    diagnostic_context: CoerceSharedDiagnosticContext,
    source: ReborrowDataField<'tcx>,
    target: ReborrowDataField<'tcx>,
) -> ErrorGuaranteed {
    tcx.dcx().emit_err(diagnostics::CoerceSharedFieldMismatch {
        span: target.span,
        source_span: source.span,
        impl_span: diagnostic_context.impl_span,
        source_name: source.name,
        source_ty: source.ty,
        target_name: target.name,
        target_ty: target.ty,
        trait_name,
    })
}

enum FieldRelation {
    Equal,
    MutRefToSharedRef,
}

// Normalizing the outer `CoerceShared` types does not normalize their fields:
// instantiating a field can expose projections. Each candidate relation uses a
// fresh inference context, so failed checks cannot affect the next one; this
// intentionally normalizes the fields for each check.
fn field_tys_satisfy_relation_after_normalization_and_resolution<'tcx>(
    tcx: TyCtxt<'tcx>,
    impl_did: LocalDefId,
    param_env: ty::ParamEnv<'tcx>,
    source_ty: Ty<'tcx>,
    target_ty: Ty<'tcx>,
    span: Span,
    relation: FieldRelation,
) -> bool {
    let infcx = tcx.infer_ctxt().build(TypingMode::non_body_analysis());
    let cause = traits::ObligationCause::misc(span, impl_did);
    let ocx = ObligationCtxt::new(&infcx);

    let Ok((source_ty, target_ty)) =
        ocx.deeply_normalize(&cause, param_env, Unnormalized::new_wip((source_ty, target_ty)))
    else {
        return false;
    };

    if !ocx.evaluate_obligations_error_on_ambiguity().is_empty() {
        return false;
    }

    match relation {
        FieldRelation::Equal => {
            if infcx.relate(param_env, source_ty, ty::Variance::Invariant, target_ty, span).is_err()
            {
                return false;
            }
        }
        FieldRelation::MutRefToSharedRef => {
            let (
                &ty::Ref(source_region, source_referent_ty, ty::Mutability::Mut),
                &ty::Ref(target_region, target_referent_ty, ty::Mutability::Not),
            ) = (source_ty.kind(), target_ty.kind())
            else {
                return false;
            };
            if source_region != target_region {
                return false;
            }
            if ocx.sup(&cause, param_env, target_referent_ty, source_referent_ty).is_err() {
                return false;
            }
        }
    };

    ocx.evaluate_obligations_error_on_ambiguity().is_empty()
        && ocx.resolve_regions(impl_did, param_env, []).is_empty()
}
