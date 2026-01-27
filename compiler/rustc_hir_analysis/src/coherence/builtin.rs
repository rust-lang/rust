//! Check properties that are required by built-in traits and set
//! up data structures required by type-checking/codegen.

use std::collections::BTreeMap;

use rustc_data_structures::fx::FxHashSet;
use rustc_errors::{ErrorGuaranteed, MultiSpan};
use rustc_hir as hir;
use rustc_hir::ItemKind;
use rustc_hir::def_id::{DefId, LocalDefId};
use rustc_hir::lang_items::LangItem;
use rustc_infer::infer::{self, InferCtxt, RegionResolutionError, SubregionOrigin, TyCtxtInferExt};
use rustc_infer::traits::{Obligation, PredicateObligations};
use rustc_middle::ty::adjustment::{CoerceSharedInfo, CoerceUnsizedInfo};
use rustc_middle::ty::print::PrintTraitRefExt as _;
use rustc_middle::ty::relate::solver_relating::RelateExt;
use rustc_middle::ty::{
    self, Ty, TyCtxt, TypeVisitableExt, TypingMode, suggest_constraining_type_params,
};
use rustc_span::{DUMMY_SP, Span, sym};
use rustc_trait_selection::error_reporting::InferCtxtErrorExt;
use rustc_trait_selection::traits::misc::{
    ConstParamTyImplementationError, CopyImplementationError, InfringingFieldsReason,
    type_allowed_to_implement_const_param_ty, type_allowed_to_implement_copy,
};
use rustc_trait_selection::traits::{self, FulfillmentError, ObligationCause, ObligationCtxt};
use tracing::debug;

use crate::errors;

pub(super) fn check_trait<'tcx>(
    tcx: TyCtxt<'tcx>,
    trait_def_id: DefId,
    impl_def_id: LocalDefId,
    impl_header: ty::ImplTraitHeader<'tcx>,
) -> Result<(), ErrorGuaranteed> {
    let lang_items = tcx.lang_items();
    let checker = Checker { tcx, trait_def_id, impl_def_id, impl_header };
    checker.check(lang_items.drop_trait(), visit_implementation_of_drop)?;
    checker.check(lang_items.async_drop_trait(), visit_implementation_of_drop)?;
    checker.check(lang_items.copy_trait(), visit_implementation_of_copy)?;
    checker.check(lang_items.const_param_ty_trait(), |checker| {
        visit_implementation_of_const_param_ty(checker)
    })?;
    checker.check(lang_items.coerce_unsized_trait(), visit_implementation_of_coerce_unsized)?;
    checker.check(lang_items.reborrow(), visit_implementation_of_reborrow)?;
    checker.check(lang_items.coerce_shared(), visit_implementation_of_coerce_shared)?;
    checker
        .check(lang_items.dispatch_from_dyn_trait(), visit_implementation_of_dispatch_from_dyn)?;
    checker.check(
        lang_items.coerce_pointee_validated_trait(),
        visit_implementation_of_coerce_pointee_validity,
    )?;
    Ok(())
}

struct Checker<'tcx> {
    tcx: TyCtxt<'tcx>,
    trait_def_id: DefId,
    impl_def_id: LocalDefId,
    impl_header: ty::ImplTraitHeader<'tcx>,
}

impl<'tcx> Checker<'tcx> {
    fn check(
        &self,
        trait_def_id: Option<DefId>,
        f: impl FnOnce(&Self) -> Result<(), ErrorGuaranteed>,
    ) -> Result<(), ErrorGuaranteed> {
        if Some(self.trait_def_id) == trait_def_id { f(self) } else { Ok(()) }
    }
}

fn visit_implementation_of_drop(checker: &Checker<'_>) -> Result<(), ErrorGuaranteed> {
    let tcx = checker.tcx;
    let impl_did = checker.impl_def_id;
    // Destructors only work on local ADT types.
    match checker.impl_header.trait_ref.instantiate_identity().self_ty().kind() {
        ty::Adt(def, _) if def.did().is_local() => return Ok(()),
        ty::Error(_) => return Ok(()),
        _ => {}
    }

    let impl_ = tcx.hir_expect_item(impl_did).expect_impl();

    Err(tcx.dcx().emit_err(errors::DropImplOnWrongItem {
        span: impl_.self_ty.span,
        trait_: tcx.item_name(checker.impl_header.trait_ref.skip_binder().def_id),
    }))
}

fn visit_implementation_of_copy(checker: &Checker<'_>) -> Result<(), ErrorGuaranteed> {
    let tcx = checker.tcx;
    let impl_header = checker.impl_header;
    let impl_did = checker.impl_def_id;
    debug!("visit_implementation_of_copy: impl_did={:?}", impl_did);

    let self_type = impl_header.trait_ref.instantiate_identity().self_ty();
    debug!("visit_implementation_of_copy: self_type={:?} (bound)", self_type);

    let param_env = tcx.param_env(impl_did);
    assert!(!self_type.has_escaping_bound_vars());

    debug!("visit_implementation_of_copy: self_type={:?} (free)", self_type);

    if let ty::ImplPolarity::Negative = impl_header.polarity {
        return Ok(());
    }

    let cause = traits::ObligationCause::misc(DUMMY_SP, impl_did);
    match type_allowed_to_implement_copy(tcx, param_env, self_type, cause, impl_header.safety) {
        Ok(()) => Ok(()),
        Err(CopyImplementationError::InfringingFields(fields)) => {
            let span = tcx.hir_expect_item(impl_did).expect_impl().self_ty.span;
            Err(infringing_fields_error(
                tcx,
                fields.into_iter().map(|(field, ty, reason)| (tcx.def_span(field.did), ty, reason)),
                LangItem::Copy,
                impl_did,
                span,
            ))
        }
        Err(CopyImplementationError::NotAnAdt) => {
            let span = tcx.hir_expect_item(impl_did).expect_impl().self_ty.span;
            Err(tcx.dcx().emit_err(errors::CopyImplOnNonAdt { span }))
        }
        Err(CopyImplementationError::HasDestructor) => {
            let span = tcx.hir_expect_item(impl_did).expect_impl().self_ty.span;
            Err(tcx.dcx().emit_err(errors::CopyImplOnTypeWithDtor { span }))
        }
        Err(CopyImplementationError::HasUnsafeFields) => {
            let span = tcx.hir_expect_item(impl_did).expect_impl().self_ty.span;
            Err(tcx
                .dcx()
                .span_delayed_bug(span, format!("cannot implement `Copy` for `{}`", self_type)))
        }
    }
}

fn visit_implementation_of_const_param_ty(checker: &Checker<'_>) -> Result<(), ErrorGuaranteed> {
    let tcx = checker.tcx;
    let header = checker.impl_header;
    let impl_did = checker.impl_def_id;
    let self_type = header.trait_ref.instantiate_identity().self_ty();
    assert!(!self_type.has_escaping_bound_vars());

    let param_env = tcx.param_env(impl_did);

    if let ty::ImplPolarity::Negative | ty::ImplPolarity::Reservation = header.polarity {
        return Ok(());
    }

    let cause = traits::ObligationCause::misc(DUMMY_SP, impl_did);
    match type_allowed_to_implement_const_param_ty(tcx, param_env, self_type, cause) {
        Ok(()) => Ok(()),
        Err(ConstParamTyImplementationError::InfrigingFields(fields)) => {
            let span = tcx.hir_expect_item(impl_did).expect_impl().self_ty.span;
            Err(infringing_fields_error(
                tcx,
                fields.into_iter().map(|(field, ty, reason)| (tcx.def_span(field.did), ty, reason)),
                LangItem::ConstParamTy,
                impl_did,
                span,
            ))
        }
        Err(ConstParamTyImplementationError::NotAnAdtOrBuiltinAllowed) => {
            let span = tcx.hir_expect_item(impl_did).expect_impl().self_ty.span;
            Err(tcx.dcx().emit_err(errors::ConstParamTyImplOnNonAdt { span }))
        }
        Err(ConstParamTyImplementationError::InvalidInnerTyOfBuiltinTy(infringing_tys)) => {
            let span = tcx.hir_expect_item(impl_did).expect_impl().self_ty.span;
            Err(infringing_fields_error(
                tcx,
                infringing_tys.into_iter().map(|(ty, reason)| (span, ty, reason)),
                LangItem::ConstParamTy,
                impl_did,
                span,
            ))
        }
        Err(ConstParamTyImplementationError::UnsizedConstParamsFeatureRequired) => {
            let span = tcx.hir_expect_item(impl_did).expect_impl().self_ty.span;
            Err(tcx.dcx().emit_err(errors::ConstParamTyImplOnUnsized { span }))
        }
    }
}

fn visit_implementation_of_coerce_unsized(checker: &Checker<'_>) -> Result<(), ErrorGuaranteed> {
    let tcx = checker.tcx;
    let impl_did = checker.impl_def_id;
    debug!("visit_implementation_of_coerce_unsized: impl_did={:?}", impl_did);

    // Just compute this for the side-effects, in particular reporting
    // errors; other parts of the code may demand it for the info of
    // course.
    tcx.ensure_ok().coerce_unsized_info(impl_did)
}

fn visit_implementation_of_reborrow(checker: &Checker<'_>) -> Result<(), ErrorGuaranteed> {
    let tcx = checker.tcx;
    let impl_did = checker.impl_def_id;
    debug!("visit_implementation_of_reborrow: impl_did={:?}", impl_did);

    // Just compute this for the side-effects, in particular reporting
    // errors; other parts of the code may demand it for the info of
    // course.
    reborrow_info(tcx, impl_did)
}

fn visit_implementation_of_coerce_shared(checker: &Checker<'_>) -> Result<(), ErrorGuaranteed> {
    let tcx = checker.tcx;
    let impl_did = checker.impl_def_id;
    debug!("visit_implementation_of_coerce_shared: impl_did={:?}", impl_did);

    // Just compute this for the side-effects, in particular reporting
    // errors; other parts of the code may demand it for the info of
    // course.
    tcx.ensure_ok().coerce_shared_info(impl_did)
}

fn is_from_coerce_pointee_derive(tcx: TyCtxt<'_>, span: Span) -> bool {
    span.ctxt()
        .outer_expn_data()
        .macro_def_id
        .is_some_and(|def_id| tcx.is_diagnostic_item(sym::CoercePointee, def_id))
}

fn visit_implementation_of_dispatch_from_dyn(checker: &Checker<'_>) -> Result<(), ErrorGuaranteed> {
    let tcx = checker.tcx;
    let impl_did = checker.impl_def_id;
    let trait_ref = checker.impl_header.trait_ref.instantiate_identity();
    debug!("visit_implementation_of_dispatch_from_dyn: impl_did={:?}", impl_did);

    let span = tcx.def_span(impl_did);
    let trait_name = "DispatchFromDyn";

    let source = trait_ref.self_ty();
    let target = {
        assert!(tcx.is_lang_item(trait_ref.def_id, LangItem::DispatchFromDyn));

        trait_ref.args.type_at(1)
    };

    // Check `CoercePointee` impl is WF -- if not, then there's no reason to report
    // redundant errors for `DispatchFromDyn`. This is best effort, though.
    let mut res = Ok(());
    tcx.for_each_relevant_impl(
        tcx.require_lang_item(LangItem::CoerceUnsized, span),
        source,
        |impl_def_id| {
            res = res.and(tcx.ensure_ok().coerce_unsized_info(impl_def_id));
        },
    );
    res?;

    debug!("visit_implementation_of_dispatch_from_dyn: {:?} -> {:?}", source, target);

    let param_env = tcx.param_env(impl_did);

    let infcx = tcx.infer_ctxt().build(TypingMode::non_body_analysis());
    let cause = ObligationCause::misc(span, impl_did);

    // Later parts of the compiler rely on all DispatchFromDyn types to be ABI-compatible with raw
    // pointers. This is enforced here: we only allow impls for references, raw pointers, and things
    // that are effectively repr(transparent) newtypes around types that already hav a
    // DispatchedFromDyn impl. We cannot literally use repr(transparent) on those types since some
    // of them support an allocator, but we ensure that for the cases where the type implements this
    // trait, they *do* satisfy the repr(transparent) rules, and then we assume that everything else
    // in the compiler (in particular, all the call ABI logic) will treat them as repr(transparent)
    // even if they do not carry that attribute.
    match (source.kind(), target.kind()) {
        (&ty::Pat(_, pat_a), &ty::Pat(_, pat_b)) => {
            if pat_a != pat_b {
                return Err(tcx.dcx().emit_err(errors::CoerceSamePatKind {
                    span,
                    trait_name,
                    pat_a: pat_a.to_string(),
                    pat_b: pat_b.to_string(),
                }));
            }
            Ok(())
        }

        (&ty::Ref(r_a, _, mutbl_a), ty::Ref(r_b, _, mutbl_b))
            if r_a == *r_b && mutbl_a == *mutbl_b =>
        {
            Ok(())
        }
        (&ty::RawPtr(_, a_mutbl), &ty::RawPtr(_, b_mutbl)) if a_mutbl == b_mutbl => Ok(()),
        (&ty::Adt(def_a, args_a), &ty::Adt(def_b, args_b))
            if def_a.is_struct() && def_b.is_struct() =>
        {
            if def_a != def_b {
                let source_path = tcx.def_path_str(def_a.did());
                let target_path = tcx.def_path_str(def_b.did());
                return Err(tcx.dcx().emit_err(errors::CoerceSameStruct {
                    span,
                    trait_name,
                    note: true,
                    source_path,
                    target_path,
                }));
            }

            if def_a.repr().c() || def_a.repr().packed() {
                return Err(tcx.dcx().emit_err(errors::DispatchFromDynRepr { span }));
            }

            let fields = &def_a.non_enum_variant().fields;

            let mut res = Ok(());
            let coerced_fields = fields
                .iter_enumerated()
                .filter_map(|(i, field)| {
                    // Ignore PhantomData fields
                    let unnormalized_ty = tcx.type_of(field.did).instantiate_identity();
                    if tcx
                        .try_normalize_erasing_regions(
                            ty::TypingEnv::non_body_analysis(tcx, def_a.did()),
                            unnormalized_ty,
                        )
                        .unwrap_or(unnormalized_ty)
                        .is_phantom_data()
                    {
                        return None;
                    }

                    let ty_a = field.ty(tcx, args_a);
                    let ty_b = field.ty(tcx, args_b);

                    // FIXME: We could do normalization here, but is it really worth it?
                    if ty_a == ty_b {
                        // Allow 1-ZSTs that don't mention type params.
                        //
                        // Allowing type params here would allow us to possibly transmute
                        // between ZSTs, which may be used to create library unsoundness.
                        if let Ok(layout) =
                            tcx.layout_of(infcx.typing_env(param_env).as_query_input(ty_a))
                            && layout.is_1zst()
                            && !ty_a.has_non_region_param()
                        {
                            // ignore 1-ZST fields
                            return None;
                        }

                        res = Err(tcx.dcx().emit_err(errors::DispatchFromDynZST {
                            span,
                            name: field.ident(tcx),
                            ty: ty_a,
                        }));

                        None
                    } else {
                        Some((i, ty_a, ty_b, tcx.def_span(field.did)))
                    }
                })
                .collect::<Vec<_>>();
            res?;

            if coerced_fields.is_empty() {
                return Err(tcx.dcx().emit_err(errors::CoerceNoField {
                    span,
                    trait_name,
                    note: true,
                }));
            } else if let &[(_, ty_a, ty_b, field_span)] = &coerced_fields[..] {
                let ocx = ObligationCtxt::new_with_diagnostics(&infcx);
                ocx.register_obligation(Obligation::new(
                    tcx,
                    cause.clone(),
                    param_env,
                    ty::TraitRef::new(tcx, trait_ref.def_id, [ty_a, ty_b]),
                ));
                let errors = ocx.evaluate_obligations_error_on_ambiguity();
                if !errors.is_empty() {
                    if is_from_coerce_pointee_derive(tcx, span) {
                        return Err(tcx.dcx().emit_err(errors::CoerceFieldValidity {
                            span,
                            trait_name,
                            ty: trait_ref.self_ty(),
                            field_span,
                            field_ty: ty_a,
                        }));
                    } else {
                        return Err(infcx.err_ctxt().report_fulfillment_errors(errors));
                    }
                }

                // Finally, resolve all regions.
                ocx.resolve_regions_and_report_errors(impl_did, param_env, [])?;

                Ok(())
            } else {
                return Err(tcx.dcx().emit_err(errors::CoerceMulti {
                    span,
                    trait_name,
                    number: coerced_fields.len(),
                    fields: coerced_fields.iter().map(|(_, _, _, s)| *s).collect::<Vec<_>>().into(),
                }));
            }
        }
        _ => Err(tcx.dcx().emit_err(errors::CoerceUnsizedNonStruct { span, trait_name })),
    }
}

fn structurally_normalize_ty<'tcx>(
    tcx: TyCtxt<'tcx>,
    infcx: &InferCtxt<'tcx>,
    impl_did: LocalDefId,
    span: Span,
    ty: Ty<'tcx>,
) -> Option<(Ty<'tcx>, PredicateObligations<'tcx>)> {
    let ocx = ObligationCtxt::new(infcx);
    let Ok(normalized_ty) = ocx.structurally_normalize_ty(
        &traits::ObligationCause::misc(span, impl_did),
        tcx.param_env(impl_did),
        ty,
    ) else {
        // We shouldn't have errors here in the old solver, except for
        // evaluate/fulfill mismatches, but that's not a reason for an ICE.
        return None;
    };
    let errors = ocx.try_evaluate_obligations();
    if !errors.is_empty() {
        if infcx.next_trait_solver() {
            unreachable!();
        }
        // We shouldn't have errors here in the old solver, except for
        // evaluate/fulfill mismatches, but that's not a reason for an ICE.
        debug!(?errors, "encountered errors while fulfilling");
        return None;
    }

    Some((normalized_ty, ocx.into_pending_obligations()))
}

pub(crate) fn reborrow_info<'tcx>(
    tcx: TyCtxt<'tcx>,
    impl_did: LocalDefId,
) -> Result<(), ErrorGuaranteed> {
    debug!("compute_reborrow_info(impl_did={:?})", impl_did);
    let infcx = tcx.infer_ctxt().build(TypingMode::non_body_analysis());
    let span = tcx.def_span(impl_did);
    let trait_name = "Reborrow";

    let reborrow_trait = tcx.require_lang_item(LangItem::Reborrow, span);

    let source = tcx.type_of(impl_did).instantiate_identity();
    let trait_ref = tcx.impl_trait_ref(impl_did).instantiate_identity();
    let lifetime_params_count = tcx
        .generics_of(impl_did)
        .own_params
        .iter()
        .filter(|p| matches!(p.kind, ty::GenericParamDefKind::Lifetime))
        .count();

    if lifetime_params_count != 1 {
        return Err(tcx
            .dcx()
            .emit_err(errors::CoerceSharedNotSingleLifetimeParam { span, trait_name }));
    }

    assert_eq!(trait_ref.def_id, reborrow_trait);
    let param_env = tcx.param_env(impl_did);
    assert!(!source.has_escaping_bound_vars());

    let (def, args) = match source.kind() {
        &ty::Adt(def, args) if def.is_struct() => (def, args),
        _ => {
            // Note: reusing error here as it takes trait_name as argument.
            return Err(tcx.dcx().emit_err(errors::CoerceUnsizedNonStruct { span, trait_name }));
        }
    };

    let lifetimes_count = args.iter().filter(|arg| arg.as_region().is_some()).count();
    let data_fields = def
        .non_enum_variant()
        .fields
        .iter()
        .filter_map(|f| {
            // Ignore PhantomData fields
            let ty = f.ty(tcx, args);
            if ty.is_phantom_data() {
                return None;
            }
            Some((ty, tcx.def_span(f.did)))
        })
        .collect::<Vec<_>>();

    if lifetimes_count != 1 {
        let item = tcx.hir_expect_item(impl_did);
        let _span = if let ItemKind::Impl(hir::Impl { of_trait: Some(of_trait), .. }) = &item.kind {
            of_trait.trait_ref.path.span
        } else {
            tcx.def_span(impl_did)
        };

        return Err(tcx.dcx().emit_err(errors::CoerceSharedMulti { span, trait_name }));
    }

    if data_fields.is_empty() {
        return Ok(());
    }

    // We've found some data fields. They must all be either be Copy or Reborrow.
    for (field, span) in data_fields {
        if assert_field_type_is_reborrow(
            tcx,
            &infcx,
            reborrow_trait,
            impl_did,
            param_env,
            field,
            span,
        )
        .is_ok()
        {
            // Field implements Reborrow.
            return Ok(());
        }

        // Field does not implement Reborrow: it must be Copy.
        assert_field_type_is_copy(tcx, &infcx, impl_did, param_env, field, span)?;
    }

    Ok(())
}

fn assert_field_type_is_reborrow<'tcx>(
    tcx: TyCtxt<'tcx>,
    infcx: &InferCtxt<'tcx>,
    reborrow_trait: DefId,
    impl_did: LocalDefId,
    param_env: ty::ParamEnv<'tcx>,
    ty: Ty<'tcx>,
    span: Span,
) -> Result<(), Vec<FulfillmentError<'tcx>>> {
    if ty.ref_mutability() == Some(ty::Mutability::Mut) {
        // Mutable references are Reborrow but not really.
        return Ok(());
    }
    let ocx = ObligationCtxt::new_with_diagnostics(infcx);
    let cause = traits::ObligationCause::misc(span, impl_did);
    let obligation =
        Obligation::new(tcx, cause, param_env, ty::TraitRef::new(tcx, reborrow_trait, [ty]));
    ocx.register_obligation(obligation);
    let errors = ocx.evaluate_obligations_error_on_ambiguity();

    if !errors.is_empty() { Err(errors) } else { Ok(()) }
}

pub(crate) fn coerce_shared_info<'tcx>(
    tcx: TyCtxt<'tcx>,
    impl_did: LocalDefId,
) -> Result<CoerceSharedInfo, ErrorGuaranteed> {
    debug!("compute_coerce_shared_info(impl_did={:?})", impl_did);
    let infcx = tcx.infer_ctxt().build(TypingMode::non_body_analysis());
    let span = tcx.def_span(impl_did);
    let trait_name = "CoerceShared";

    let coerce_shared_trait = tcx.require_lang_item(LangItem::CoerceShared, span);

    let source = tcx.type_of(impl_did).instantiate_identity();
    let trait_ref = tcx.impl_trait_ref(impl_did).instantiate_identity();
    let lifetime_params_count = tcx
        .generics_of(impl_did)
        .own_params
        .iter()
        .filter(|p| matches!(p.kind, ty::GenericParamDefKind::Lifetime))
        .count();

    if lifetime_params_count != 1 {
        return Err(tcx
            .dcx()
            .emit_err(errors::CoerceSharedNotSingleLifetimeParam { span, trait_name }));
    }

    assert_eq!(trait_ref.def_id, coerce_shared_trait);
    let Some((target, _obligations)) =
        structurally_normalize_ty(tcx, &infcx, impl_did, span, trait_ref.args.type_at(1))
    else {
        todo!("something went wrong with structurally_normalize_ty");
    };

    let param_env = tcx.param_env(impl_did);
    assert!(!source.has_escaping_bound_vars());

    let data = match (source.kind(), target.kind()) {
        (&ty::Adt(def_a, args_a), &ty::Adt(def_b, args_b))
            if def_a.is_struct() && def_b.is_struct() =>
        {
            // Check that both A and B have exactly one lifetime argument, and that they have the
            // same number of data fields that is not more than 1. The eventual intention is to
            // support multiple lifetime arguments (with the reborrowed lifetimes inferred from
            // usage one way or another) and multiple data fields with B allowed to leave out fields
            // from A. The current state is just the simplest choice.
            let a_lifetimes_count = args_a.iter().filter(|arg| arg.as_region().is_some()).count();
            let a_data_fields = def_a
                .non_enum_variant()
                .fields
                .iter_enumerated()
                .filter_map(|(i, f)| {
                    let a = f.ty(tcx, args_b);

                    if a.is_phantom_data() {
                        return None;
                    }

                    Some((i, a, tcx.def_span(f.did)))
                })
                .collect::<Vec<_>>();
            let b_lifetimes_count = args_b.iter().filter(|arg| arg.as_region().is_some()).count();
            let b_data_fields = def_b
                .non_enum_variant()
                .fields
                .iter_enumerated()
                .filter_map(|(i, f)| {
                    let b = f.ty(tcx, args_b);

                    if b.is_phantom_data() {
                        return None;
                    }

                    Some((i, b, tcx.def_span(f.did)))
                })
                .collect::<Vec<_>>();

            if a_lifetimes_count != 1
                || b_lifetimes_count != 1
                || a_data_fields.len() > 1
                || b_data_fields.len() > 1
                || a_data_fields.len() != b_data_fields.len()
            {
                let item = tcx.hir_expect_item(impl_did);
                let span = if let ItemKind::Impl(hir::Impl { of_trait: Some(of_trait), .. }) =
                    &item.kind
                {
                    of_trait.trait_ref.path.span
                } else {
                    tcx.def_span(impl_did)
                };

                return Err(tcx.dcx().emit_err(errors::CoerceSharedMulti { span, trait_name }));
            }

            let kind = ty::adjustment::CoerceSharedInfo {};
            if a_data_fields.len() == 1 {
                // We found one data field for both: we'll attempt to perform CoerceShared between
                // them below.
                let (_a_i, a, span_a) = a_data_fields[0];
                let (_b_i, b, span_b) = b_data_fields[0];

                Some((a, b, coerce_shared_trait, kind, span_a, span_b))
            } else {
                // We found no data fields in either: this is a reborrowable marker type being
                // coerced into a shared marker. That is fine too.
                None
            }
        }

        _ => {
            // Note: reusing CoerceUnsizedNonStruct error as it takes trait_name as argument.
            return Err(tcx.dcx().emit_err(errors::CoerceUnsizedNonStruct { span, trait_name }));
        }
    };

    // We've proven that we have two types with one lifetime each and 0 or 1 data fields each.
    if let Some((source, target, trait_def_id, _kind, source_field_span, _target_field_span)) = data
    {
        // struct Source(SourceData);
        // struct Target(TargetData);
        //
        // 1 data field each; they must be the same type and Copy, or relate to one another using
        // CoerceShared.
        if source.ref_mutability() == Some(ty::Mutability::Mut)
            && target.ref_mutability() == Some(ty::Mutability::Not)
            && infcx
                .eq_structurally_relating_aliases(
                    param_env,
                    source.peel_refs(),
                    target.peel_refs(),
                    source_field_span,
                )
                .is_ok()
        {
            // &mut T implements CoerceShared to &T, except not really.
            return Ok(CoerceSharedInfo {});
        }
        if infcx
            .eq_structurally_relating_aliases(param_env, source, target, source_field_span)
            .is_err()
        {
            // The two data fields don't agree on a common type; this means
            // that they must be `A: CoerceShared<B>`. Register an obligation
            // for that.
            let ocx = ObligationCtxt::new_with_diagnostics(&infcx);
            let cause = traits::ObligationCause::misc(span, impl_did);
            let obligation = Obligation::new(
                tcx,
                cause,
                param_env,
                ty::TraitRef::new(tcx, trait_def_id, [source, target]),
            );
            ocx.register_obligation(obligation);
            let errors = ocx.evaluate_obligations_error_on_ambiguity();

            if !errors.is_empty() {
                return Err(infcx.err_ctxt().report_fulfillment_errors(errors));
            }
            // Finally, resolve all regions.
            ocx.resolve_regions_and_report_errors(impl_did, param_env, [])?;
        } else {
            // Types match: check that it is Copy.
            assert_field_type_is_copy(tcx, &infcx, impl_did, param_env, source, source_field_span)?;
        }
    }

    Ok(CoerceSharedInfo {})
}

fn assert_field_type_is_copy<'tcx>(
    tcx: TyCtxt<'tcx>,
    infcx: &InferCtxt<'tcx>,
    impl_did: LocalDefId,
    param_env: ty::ParamEnv<'tcx>,
    ty: Ty<'tcx>,
    span: Span,
) -> Result<(), ErrorGuaranteed> {
    let copy_trait = tcx.require_lang_item(LangItem::Copy, span);
    let ocx = ObligationCtxt::new_with_diagnostics(infcx);
    let cause = traits::ObligationCause::misc(span, impl_did);
    let obligation =
        Obligation::new(tcx, cause, param_env, ty::TraitRef::new(tcx, copy_trait, [ty]));
    ocx.register_obligation(obligation);
    let errors = ocx.evaluate_obligations_error_on_ambiguity();

    if !errors.is_empty() {
        Err(infcx.err_ctxt().report_fulfillment_errors(errors))
    } else {
        Ok(())
    }
}

pub(crate) fn coerce_unsized_info<'tcx>(
    tcx: TyCtxt<'tcx>,
    impl_did: LocalDefId,
) -> Result<CoerceUnsizedInfo, ErrorGuaranteed> {
    debug!("compute_coerce_unsized_info(impl_did={:?})", impl_did);
    let span = tcx.def_span(impl_did);
    let trait_name = "CoerceUnsized";

    let coerce_unsized_trait = tcx.require_lang_item(LangItem::CoerceUnsized, span);
    let unsize_trait = tcx.require_lang_item(LangItem::Unsize, span);

    let source = tcx.type_of(impl_did).instantiate_identity();
    let trait_ref = tcx.impl_trait_ref(impl_did).instantiate_identity();

    assert_eq!(trait_ref.def_id, coerce_unsized_trait);
    let target = trait_ref.args.type_at(1);
    debug!("visit_implementation_of_coerce_unsized: {:?} -> {:?} (bound)", source, target);

    let param_env = tcx.param_env(impl_did);
    assert!(!source.has_escaping_bound_vars());

    debug!("visit_implementation_of_coerce_unsized: {:?} -> {:?} (free)", source, target);

    let infcx = tcx.infer_ctxt().build(TypingMode::non_body_analysis());
    let cause = ObligationCause::misc(span, impl_did);
    let check_mutbl = |mt_a: ty::TypeAndMut<'tcx>,
                       mt_b: ty::TypeAndMut<'tcx>,
                       mk_ptr: &dyn Fn(Ty<'tcx>) -> Ty<'tcx>| {
        if mt_a.mutbl < mt_b.mutbl {
            infcx
                .err_ctxt()
                .report_mismatched_types(
                    &cause,
                    param_env,
                    mk_ptr(mt_b.ty),
                    target,
                    ty::error::TypeError::Mutability,
                )
                .emit();
        }
        (mt_a.ty, mt_b.ty, unsize_trait, None, span)
    };
    let (source, target, trait_def_id, kind, field_span) = match (source.kind(), target.kind()) {
        (&ty::Pat(ty_a, pat_a), &ty::Pat(ty_b, pat_b)) => {
            if pat_a != pat_b {
                return Err(tcx.dcx().emit_err(errors::CoerceSamePatKind {
                    span,
                    trait_name,
                    pat_a: pat_a.to_string(),
                    pat_b: pat_b.to_string(),
                }));
            }
            (ty_a, ty_b, coerce_unsized_trait, None, span)
        }

        (&ty::Ref(r_a, ty_a, mutbl_a), &ty::Ref(r_b, ty_b, mutbl_b)) => {
            infcx.sub_regions(SubregionOrigin::RelateObjectBound(span), r_b, r_a);
            let mt_a = ty::TypeAndMut { ty: ty_a, mutbl: mutbl_a };
            let mt_b = ty::TypeAndMut { ty: ty_b, mutbl: mutbl_b };
            check_mutbl(mt_a, mt_b, &|ty| Ty::new_imm_ref(tcx, r_b, ty))
        }

        (&ty::Ref(_, ty_a, mutbl_a), &ty::RawPtr(ty_b, mutbl_b))
        | (&ty::RawPtr(ty_a, mutbl_a), &ty::RawPtr(ty_b, mutbl_b)) => {
            let mt_a = ty::TypeAndMut { ty: ty_a, mutbl: mutbl_a };
            let mt_b = ty::TypeAndMut { ty: ty_b, mutbl: mutbl_b };
            check_mutbl(mt_a, mt_b, &|ty| Ty::new_imm_ptr(tcx, ty))
        }

        (&ty::Adt(def_a, args_a), &ty::Adt(def_b, args_b))
            if def_a.is_struct() && def_b.is_struct() =>
        {
            if def_a != def_b {
                let source_path = tcx.def_path_str(def_a.did());
                let target_path = tcx.def_path_str(def_b.did());
                return Err(tcx.dcx().emit_err(errors::CoerceSameStruct {
                    span,
                    trait_name,
                    note: true,
                    source_path,
                    target_path,
                }));
            }

            // Here we are considering a case of converting
            // `S<P0...Pn>` to `S<Q0...Qn>`. As an example, let's imagine a struct `Foo<T, U>`,
            // which acts like a pointer to `U`, but carries along some extra data of type `T`:
            //
            //     struct Foo<T, U> {
            //         extra: T,
            //         ptr: *mut U,
            //     }
            //
            // We might have an impl that allows (e.g.) `Foo<T, [i32; 3]>` to be unsized
            // to `Foo<T, [i32]>`. That impl would look like:
            //
            //   impl<T, U: Unsize<V>, V> CoerceUnsized<Foo<T, V>> for Foo<T, U> {}
            //
            // Here `U = [i32; 3]` and `V = [i32]`. At runtime,
            // when this coercion occurs, we would be changing the
            // field `ptr` from a thin pointer of type `*mut [i32;
            // 3]` to a wide pointer of type `*mut [i32]` (with
            // extra data `3`). **The purpose of this check is to
            // make sure that we know how to do this conversion.**
            //
            // To check if this impl is legal, we would walk down
            // the fields of `Foo` and consider their types with
            // both generic parameters. We are looking to find that
            // exactly one (non-phantom) field has changed its
            // type, which we will expect to be the pointer that
            // is becoming fat (we could probably generalize this
            // to multiple thin pointers of the same type becoming
            // fat, but we don't). In this case:
            //
            // - `extra` has type `T` before and type `T` after
            // - `ptr` has type `*mut U` before and type `*mut V` after
            //
            // Since just one field changed, we would then check
            // that `*mut U: CoerceUnsized<*mut V>` is implemented
            // (in other words, that we know how to do this
            // conversion). This will work out because `U:
            // Unsize<V>`, and we have a builtin rule that `*mut
            // U` can be coerced to `*mut V` if `U: Unsize<V>`.
            let fields = &def_a.non_enum_variant().fields;
            let diff_fields = fields
                .iter_enumerated()
                .filter_map(|(i, f)| {
                    let (a, b) = (f.ty(tcx, args_a), f.ty(tcx, args_b));

                    // Ignore PhantomData fields
                    let unnormalized_ty = tcx.type_of(f.did).instantiate_identity();
                    if tcx
                        .try_normalize_erasing_regions(
                            ty::TypingEnv::non_body_analysis(tcx, def_a.did()),
                            unnormalized_ty,
                        )
                        .unwrap_or(unnormalized_ty)
                        .is_phantom_data()
                    {
                        return None;
                    }

                    // Ignore fields that aren't changed; it may
                    // be that we could get away with subtyping or
                    // something more accepting, but we use
                    // equality because we want to be able to
                    // perform this check without computing
                    // variance or constraining opaque types' hidden types.
                    // (This is because we may have to evaluate constraint
                    // expressions in the course of execution.)
                    // See e.g., #41936.
                    if a == b {
                        return None;
                    }

                    // Collect up all fields that were significantly changed
                    // i.e., those that contain T in coerce_unsized T -> U
                    Some((i, a, b, tcx.def_span(f.did)))
                })
                .collect::<Vec<_>>();

            if diff_fields.is_empty() {
                return Err(tcx.dcx().emit_err(errors::CoerceNoField {
                    span,
                    trait_name,
                    note: true,
                }));
            } else if diff_fields.len() > 1 {
                let item = tcx.hir_expect_item(impl_did);
                let span = if let ItemKind::Impl(hir::Impl { of_trait: Some(of_trait), .. }) =
                    &item.kind
                {
                    of_trait.trait_ref.path.span
                } else {
                    tcx.def_span(impl_did)
                };

                return Err(tcx.dcx().emit_err(errors::CoerceMulti {
                    span,
                    trait_name,
                    number: diff_fields.len(),
                    fields: diff_fields.iter().map(|(_, _, _, s)| *s).collect::<Vec<_>>().into(),
                }));
            }

            let (i, a, b, field_span) = diff_fields[0];
            let kind = ty::adjustment::CustomCoerceUnsized::Struct(i);
            (a, b, coerce_unsized_trait, Some(kind), field_span)
        }

        _ => {
            return Err(tcx.dcx().emit_err(errors::CoerceUnsizedNonStruct { span, trait_name }));
        }
    };

    // Register an obligation for `A: Trait<B>`.
    let ocx = ObligationCtxt::new_with_diagnostics(&infcx);
    let cause = traits::ObligationCause::misc(span, impl_did);
    let obligation = Obligation::new(
        tcx,
        cause,
        param_env,
        ty::TraitRef::new(tcx, trait_def_id, [source, target]),
    );
    ocx.register_obligation(obligation);
    let errors = ocx.evaluate_obligations_error_on_ambiguity();

    if !errors.is_empty() {
        if is_from_coerce_pointee_derive(tcx, span) {
            return Err(tcx.dcx().emit_err(errors::CoerceFieldValidity {
                span,
                trait_name,
                ty: trait_ref.self_ty(),
                field_span,
                field_ty: source,
            }));
        } else {
            return Err(infcx.err_ctxt().report_fulfillment_errors(errors));
        }
    }

    // Finally, resolve all regions.
    ocx.resolve_regions_and_report_errors(impl_did, param_env, [])?;

    Ok(CoerceUnsizedInfo { custom_kind: kind })
}

fn infringing_fields_error<'tcx>(
    tcx: TyCtxt<'tcx>,
    infringing_tys: impl Iterator<Item = (Span, Ty<'tcx>, InfringingFieldsReason<'tcx>)>,
    lang_item: LangItem,
    impl_did: LocalDefId,
    impl_span: Span,
) -> ErrorGuaranteed {
    let trait_did = tcx.require_lang_item(lang_item, impl_span);

    let trait_name = tcx.def_path_str(trait_did);

    // We'll try to suggest constraining type parameters to fulfill the requirements of
    // their `Copy` implementation.
    let mut errors: BTreeMap<_, Vec<_>> = Default::default();
    let mut bounds = vec![];

    let mut seen_tys = FxHashSet::default();

    let mut label_spans = Vec::new();

    for (span, ty, reason) in infringing_tys {
        // Only report an error once per type.
        if !seen_tys.insert(ty) {
            continue;
        }

        label_spans.push(span);

        match reason {
            InfringingFieldsReason::Fulfill(fulfillment_errors) => {
                for error in fulfillment_errors {
                    let error_predicate = error.obligation.predicate;
                    // Only note if it's not the root obligation, otherwise it's trivial and
                    // should be self-explanatory (i.e. a field literally doesn't implement Copy).

                    // FIXME: This error could be more descriptive, especially if the error_predicate
                    // contains a foreign type or if it's a deeply nested type...
                    if error_predicate != error.root_obligation.predicate {
                        errors
                            .entry((ty.to_string(), error_predicate.to_string()))
                            .or_default()
                            .push(error.obligation.cause.span);
                    }
                    if let ty::PredicateKind::Clause(ty::ClauseKind::Trait(ty::TraitPredicate {
                        trait_ref,
                        polarity: ty::PredicatePolarity::Positive,
                        ..
                    })) = error_predicate.kind().skip_binder()
                    {
                        let ty = trait_ref.self_ty();
                        if let ty::Param(_) = ty.kind() {
                            bounds.push((
                                format!("{ty}"),
                                trait_ref.print_trait_sugared().to_string(),
                                Some(trait_ref.def_id),
                            ));
                        }
                    }
                }
            }
            InfringingFieldsReason::Regions(region_errors) => {
                for error in region_errors {
                    let ty = ty.to_string();
                    match error {
                        RegionResolutionError::ConcreteFailure(origin, a, b) => {
                            let predicate = format!("{b}: {a}");
                            errors
                                .entry((ty.clone(), predicate.clone()))
                                .or_default()
                                .push(origin.span());
                            if let ty::RegionKind::ReEarlyParam(ebr) = b.kind()
                                && ebr.is_named()
                            {
                                bounds.push((b.to_string(), a.to_string(), None));
                            }
                        }
                        RegionResolutionError::GenericBoundFailure(origin, a, b) => {
                            let predicate = format!("{a}: {b}");
                            errors
                                .entry((ty.clone(), predicate.clone()))
                                .or_default()
                                .push(origin.span());
                            if let infer::region_constraints::GenericKind::Param(_) = a {
                                bounds.push((a.to_string(), b.to_string(), None));
                            }
                        }
                        _ => continue,
                    }
                }
            }
        }
    }
    let mut notes = Vec::new();
    for ((ty, error_predicate), spans) in errors {
        let span: MultiSpan = spans.into();
        notes.push(errors::ImplForTyRequires {
            span,
            error_predicate,
            trait_name: trait_name.clone(),
            ty,
        });
    }

    let mut err = tcx.dcx().create_err(errors::TraitCannotImplForTy {
        span: impl_span,
        trait_name,
        label_spans,
        notes,
    });

    suggest_constraining_type_params(
        tcx,
        tcx.hir_get_generics(impl_did).expect("impls always have generics"),
        &mut err,
        bounds
            .iter()
            .map(|(param, constraint, def_id)| (param.as_str(), constraint.as_str(), *def_id)),
        None,
    );

    err.emit()
}

fn visit_implementation_of_coerce_pointee_validity(
    checker: &Checker<'_>,
) -> Result<(), ErrorGuaranteed> {
    let tcx = checker.tcx;
    let self_ty = tcx.impl_trait_ref(checker.impl_def_id).instantiate_identity().self_ty();
    let span = tcx.def_span(checker.impl_def_id);
    if !tcx.is_builtin_derived(checker.impl_def_id.into()) {
        return Err(tcx.dcx().emit_err(errors::CoercePointeeNoUserValidityAssertion { span }));
    }
    let ty::Adt(def, _args) = self_ty.kind() else {
        return Err(tcx.dcx().emit_err(errors::CoercePointeeNotConcreteType { span }));
    };
    let did = def.did();
    // Now get a more precise span of the `struct`.
    let span = tcx.def_span(did);
    if !def.is_struct() {
        return Err(tcx
            .dcx()
            .emit_err(errors::CoercePointeeNotStruct { span, kind: def.descr().into() }));
    }
    if !def.repr().transparent() {
        return Err(tcx.dcx().emit_err(errors::CoercePointeeNotTransparent { span }));
    }
    if def.all_fields().next().is_none() {
        return Err(tcx.dcx().emit_err(errors::CoercePointeeNoField { span }));
    }
    Ok(())
}
