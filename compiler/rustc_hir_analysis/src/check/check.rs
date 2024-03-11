use crate::check::intrinsicck::InlineAsmCtxt;
use crate::errors::LinkageType;

use super::compare_impl_item::check_type_bounds;
use super::compare_impl_item::{compare_impl_method, compare_impl_ty};
use super::*;
use rustc_attr as attr;
use rustc_errors::{codes::*, MultiSpan};
use rustc_hir as hir;
use rustc_hir::def::{CtorKind, DefKind};
use rustc_hir::Node;
use rustc_infer::infer::{RegionVariableOrigin, TyCtxtInferExt};
use rustc_infer::traits::{Obligation, TraitEngineExt as _};
use rustc_lint_defs::builtin::REPR_TRANSPARENT_EXTERNAL_PRIVATE_FIELDS;
use rustc_middle::middle::stability::EvalResult;
use rustc_middle::traits::{DefiningAnchor, ObligationCauseCode};
use rustc_middle::ty::fold::BottomUpFolder;
use rustc_middle::ty::layout::{LayoutError, MAX_SIMD_LANES};
use rustc_middle::ty::util::{Discr, InspectCoroutineFields, IntTypeExt};
use rustc_middle::ty::GenericArgKind;
use rustc_middle::ty::{
    AdtDef, ParamEnv, RegionKind, TypeSuperVisitable, TypeVisitable, TypeVisitableExt,
};
use rustc_session::lint::builtin::{UNINHABITED_STATIC, UNSUPPORTED_CALLING_CONVENTIONS};
use rustc_span::symbol::sym;
use rustc_target::abi::FieldIdx;
use rustc_trait_selection::traits::error_reporting::on_unimplemented::OnUnimplementedDirective;
use rustc_trait_selection::traits::error_reporting::TypeErrCtxtExt as _;
use rustc_trait_selection::traits::outlives_bounds::InferCtxtExt as _;
use rustc_trait_selection::traits::{self, TraitEngine, TraitEngineExt as _};
use rustc_type_ir::fold::TypeFoldable;

use std::cell::LazyCell;
use std::ops::ControlFlow;

pub fn check_abi(tcx: TyCtxt<'_>, hir_id: hir::HirId, span: Span, abi: Abi) {
    match tcx.sess.target.is_abi_supported(abi) {
        Some(true) => (),
        Some(false) => {
            struct_span_code_err!(
                tcx.dcx(),
                span,
                E0570,
                "`{abi}` is not a supported ABI for the current target",
            )
            .emit();
        }
        None => {
            tcx.node_span_lint(
                UNSUPPORTED_CALLING_CONVENTIONS,
                hir_id,
                span,
                "use of calling convention not supported on this target",
                |_| {},
            );
        }
    }

    // This ABI is only allowed on function pointers
    if abi == Abi::CCmseNonSecureCall {
        struct_span_code_err!(
            tcx.dcx(),
            span,
            E0781,
            "the `\"C-cmse-nonsecure-call\"` ABI is only allowed on function pointers"
        )
        .emit();
    }
}

fn check_struct(tcx: TyCtxt<'_>, def_id: LocalDefId) {
    let def = tcx.adt_def(def_id);
    let span = tcx.def_span(def_id);
    def.destructor(tcx); // force the destructor to be evaluated

    if def.repr().simd() {
        check_simd(tcx, span, def_id);
    }

    check_transparent(tcx, def);
    check_packed(tcx, span, def);
    check_unnamed_fields(tcx, def);
}

fn check_union(tcx: TyCtxt<'_>, def_id: LocalDefId) {
    let def = tcx.adt_def(def_id);
    let span = tcx.def_span(def_id);
    def.destructor(tcx); // force the destructor to be evaluated
    check_transparent(tcx, def);
    check_union_fields(tcx, span, def_id);
    check_packed(tcx, span, def);
    check_unnamed_fields(tcx, def);
}

/// Check the representation of adts with unnamed fields.
fn check_unnamed_fields(tcx: TyCtxt<'_>, def: ty::AdtDef<'_>) {
    if def.is_enum() {
        return;
    }
    let variant = def.non_enum_variant();
    if !variant.has_unnamed_fields() {
        return;
    }
    if !def.is_anonymous() {
        let adt_kind = def.descr();
        let span = tcx.def_span(def.did());
        let unnamed_fields = variant
            .fields
            .iter()
            .filter(|f| f.is_unnamed())
            .map(|f| {
                let span = tcx.def_span(f.did);
                errors::UnnamedFieldsReprFieldDefined { span }
            })
            .collect::<Vec<_>>();
        debug_assert_ne!(unnamed_fields.len(), 0, "expect unnamed fields in this adt");
        let adt_name = tcx.item_name(def.did());
        if !def.repr().c() {
            tcx.dcx().emit_err(errors::UnnamedFieldsRepr::MissingReprC {
                span,
                adt_kind,
                adt_name,
                unnamed_fields,
                sugg_span: span.shrink_to_lo(),
            });
        }
    }
    for field in variant.fields.iter().filter(|f| f.is_unnamed()) {
        let field_ty = tcx.type_of(field.did).instantiate_identity();
        if let Some(adt) = field_ty.ty_adt_def()
            && !adt.is_enum()
        {
            if !adt.is_anonymous() && !adt.repr().c() {
                let field_ty_span = tcx.def_span(adt.did());
                tcx.dcx().emit_err(errors::UnnamedFieldsRepr::FieldMissingReprC {
                    span: tcx.def_span(field.did),
                    field_ty_span,
                    field_ty,
                    field_adt_kind: adt.descr(),
                    sugg_span: field_ty_span.shrink_to_lo(),
                });
            }
        } else {
            tcx.dcx().emit_err(errors::InvalidUnnamedFieldTy { span: tcx.def_span(field.did) });
        }
    }
}

/// Check that the fields of the `union` do not need dropping.
fn check_union_fields(tcx: TyCtxt<'_>, span: Span, item_def_id: LocalDefId) -> bool {
    let item_type = tcx.type_of(item_def_id).instantiate_identity();
    if let ty::Adt(def, args) = item_type.kind() {
        assert!(def.is_union());

        fn allowed_union_field<'tcx>(
            ty: Ty<'tcx>,
            tcx: TyCtxt<'tcx>,
            param_env: ty::ParamEnv<'tcx>,
        ) -> bool {
            // We don't just accept all !needs_drop fields, due to semver concerns.
            match ty.kind() {
                ty::Ref(..) => true, // references never drop (even mutable refs, which are non-Copy and hence fail the later check)
                ty::Tuple(tys) => {
                    // allow tuples of allowed types
                    tys.iter().all(|ty| allowed_union_field(ty, tcx, param_env))
                }
                ty::Array(elem, _len) => {
                    // Like `Copy`, we do *not* special-case length 0.
                    allowed_union_field(*elem, tcx, param_env)
                }
                _ => {
                    // Fallback case: allow `ManuallyDrop` and things that are `Copy`,
                    // also no need to report an error if the type is unresolved.
                    ty.ty_adt_def().is_some_and(|adt_def| adt_def.is_manually_drop())
                        || ty.is_copy_modulo_regions(tcx, param_env)
                        || ty.references_error()
                }
            }
        }

        let param_env = tcx.param_env(item_def_id);
        for field in &def.non_enum_variant().fields {
            let Ok(field_ty) = tcx.try_normalize_erasing_regions(param_env, field.ty(tcx, args))
            else {
                tcx.dcx().span_delayed_bug(span, "could not normalize field type");
                continue;
            };

            if !allowed_union_field(field_ty, tcx, param_env) {
                let (field_span, ty_span) = match tcx.hir().get_if_local(field.did) {
                    // We are currently checking the type this field came from, so it must be local.
                    Some(Node::Field(field)) => (field.span, field.ty.span),
                    _ => unreachable!("mir field has to correspond to hir field"),
                };
                tcx.dcx().emit_err(errors::InvalidUnionField {
                    field_span,
                    sugg: errors::InvalidUnionFieldSuggestion {
                        lo: ty_span.shrink_to_lo(),
                        hi: ty_span.shrink_to_hi(),
                    },
                    note: (),
                });
                return false;
            } else if field_ty.needs_drop(tcx, param_env) {
                // This should never happen. But we can get here e.g. in case of name resolution errors.
                tcx.dcx()
                    .span_delayed_bug(span, "we should never accept maybe-dropping union fields");
            }
        }
    } else {
        span_bug!(span, "unions must be ty::Adt, but got {:?}", item_type.kind());
    }
    true
}

/// Check that a `static` is inhabited.
fn check_static_inhabited(tcx: TyCtxt<'_>, def_id: LocalDefId) {
    // Make sure statics are inhabited.
    // Other parts of the compiler assume that there are no uninhabited places. In principle it
    // would be enough to check this for `extern` statics, as statics with an initializer will
    // have UB during initialization if they are uninhabited, but there also seems to be no good
    // reason to allow any statics to be uninhabited.
    let ty = tcx.type_of(def_id).instantiate_identity();
    let span = tcx.def_span(def_id);
    let layout = match tcx.layout_of(ParamEnv::reveal_all().and(ty)) {
        Ok(l) => l,
        // Foreign statics that overflow their allowed size should emit an error
        Err(LayoutError::SizeOverflow(_))
            if matches!(tcx.def_kind(def_id), DefKind::Static{ .. }
                if tcx.def_kind(tcx.local_parent(def_id)) == DefKind::ForeignMod) =>
        {
            tcx.dcx().emit_err(errors::TooLargeStatic { span });
            return;
        }
        // Generic statics are rejected, but we still reach this case.
        Err(e) => {
            tcx.dcx().span_delayed_bug(span, format!("{e:?}"));
            return;
        }
    };
    if layout.abi.is_uninhabited() {
        tcx.node_span_lint(
            UNINHABITED_STATIC,
            tcx.local_def_id_to_hir_id(def_id),
            span,
            "static of uninhabited type",
            |lint| {
                lint
                .note("uninhabited statics cannot be initialized, and any access would be an immediate error");
            },
        );
    }
}

/// Checks that an opaque type does not contain cycles and does not use `Self` or `T::Foo`
/// projections that would result in "inheriting lifetimes".
fn check_opaque(tcx: TyCtxt<'_>, def_id: LocalDefId) {
    let item = tcx.hir().expect_item(def_id);
    let hir::ItemKind::OpaqueTy(hir::OpaqueTy { origin, .. }) = item.kind else {
        tcx.dcx().span_bug(item.span, "expected opaque item");
    };

    // HACK(jynelson): trying to infer the type of `impl trait` breaks documenting
    // `async-std` (and `pub async fn` in general).
    // Since rustdoc doesn't care about the concrete type behind `impl Trait`, just don't look at it!
    // See https://github.com/rust-lang/rust/issues/75100
    if tcx.sess.opts.actually_rustdoc {
        return;
    }

    let span = tcx.def_span(item.owner_id.def_id);

    if tcx.type_of(item.owner_id.def_id).instantiate_identity().references_error() {
        return;
    }
    if check_opaque_for_cycles(tcx, item.owner_id.def_id, span).is_err() {
        return;
    }

    let _ = check_opaque_meets_bounds(tcx, item.owner_id.def_id, span, origin);
}

/// Checks that an opaque type does not contain cycles.
pub(super) fn check_opaque_for_cycles<'tcx>(
    tcx: TyCtxt<'tcx>,
    def_id: LocalDefId,
    span: Span,
) -> Result<(), ErrorGuaranteed> {
    let args = GenericArgs::identity_for_item(tcx, def_id);

    // First, try to look at any opaque expansion cycles, considering coroutine fields
    // (even though these aren't necessarily true errors).
    if tcx
        .try_expand_impl_trait_type(def_id.to_def_id(), args, InspectCoroutineFields::Yes)
        .is_err()
    {
        // Look for true opaque expansion cycles, but ignore coroutines.
        // This will give us any true errors. Coroutines are only problematic
        // if they cause layout computation errors.
        if tcx
            .try_expand_impl_trait_type(def_id.to_def_id(), args, InspectCoroutineFields::No)
            .is_err()
        {
            let reported = opaque_type_cycle_error(tcx, def_id, span);
            return Err(reported);
        }

        // And also look for cycle errors in the layout of coroutines.
        if let Err(&LayoutError::Cycle(guar)) =
            tcx.layout_of(tcx.param_env(def_id).and(Ty::new_opaque(tcx, def_id.to_def_id(), args)))
        {
            return Err(guar);
        }
    }

    Ok(())
}

/// Check that the concrete type behind `impl Trait` actually implements `Trait`.
///
/// This is mostly checked at the places that specify the opaque type, but we
/// check those cases in the `param_env` of that function, which may have
/// bounds not on this opaque type:
///
/// ```ignore (illustrative)
/// type X<T> = impl Clone;
/// fn f<T: Clone>(t: T) -> X<T> {
///     t
/// }
/// ```
///
/// Without this check the above code is incorrectly accepted: we would ICE if
/// some tried, for example, to clone an `Option<X<&mut ()>>`.
#[instrument(level = "debug", skip(tcx))]
fn check_opaque_meets_bounds<'tcx>(
    tcx: TyCtxt<'tcx>,
    def_id: LocalDefId,
    span: Span,
    origin: &hir::OpaqueTyOrigin,
) -> Result<(), ErrorGuaranteed> {
    let defining_use_anchor = match *origin {
        hir::OpaqueTyOrigin::FnReturn(did)
        | hir::OpaqueTyOrigin::AsyncFn(did)
        | hir::OpaqueTyOrigin::TyAlias { parent: did, .. } => did,
    };
    let param_env = tcx.param_env(defining_use_anchor);

    let infcx = tcx
        .infer_ctxt()
        .with_opaque_type_inference(DefiningAnchor::bind(tcx, defining_use_anchor))
        .build();
    let ocx = ObligationCtxt::new(&infcx);

    let args = match *origin {
        hir::OpaqueTyOrigin::FnReturn(parent)
        | hir::OpaqueTyOrigin::AsyncFn(parent)
        | hir::OpaqueTyOrigin::TyAlias { parent, .. } => GenericArgs::identity_for_item(
            tcx, parent,
        )
        .extend_to(tcx, def_id.to_def_id(), |param, _| {
            tcx.map_opaque_lifetime_to_parent_lifetime(param.def_id.expect_local()).into()
        }),
    };

    let opaque_ty = Ty::new_opaque(tcx, def_id.to_def_id(), args);

    // `ReErased` regions appear in the "parent_args" of closures/coroutines.
    // We're ignoring them here and replacing them with fresh region variables.
    // See tests in ui/type-alias-impl-trait/closure_{parent_args,wf_outlives}.rs.
    //
    // FIXME: Consider wrapping the hidden type in an existential `Binder` and instantiating it
    // here rather than using ReErased.
    let hidden_ty = tcx.type_of(def_id.to_def_id()).instantiate(tcx, args);
    let hidden_ty = tcx.fold_regions(hidden_ty, |re, _dbi| match re.kind() {
        ty::ReErased => infcx.next_region_var(RegionVariableOrigin::MiscVariable(span)),
        _ => re,
    });

    let misc_cause = traits::ObligationCause::misc(span, def_id);

    match ocx.eq(&misc_cause, param_env, opaque_ty, hidden_ty) {
        Ok(()) => {}
        Err(ty_err) => {
            let ty_err = ty_err.to_string(tcx);
            tcx.dcx().span_bug(
                span,
                format!("could not unify `{hidden_ty}` with revealed type:\n{ty_err}"),
            );
        }
    }

    // Additionally require the hidden type to be well-formed with only the generics of the opaque type.
    // Defining use functions may have more bounds than the opaque type, which is ok, as long as the
    // hidden type is well formed even without those bounds.
    let predicate =
        ty::Binder::dummy(ty::PredicateKind::Clause(ty::ClauseKind::WellFormed(hidden_ty.into())));
    ocx.register_obligation(Obligation::new(tcx, misc_cause.clone(), param_env, predicate));

    // Check that all obligations are satisfied by the implementation's
    // version.
    let errors = ocx.select_all_or_error();
    if !errors.is_empty() {
        let guar = infcx.err_ctxt().report_fulfillment_errors(errors);
        return Err(guar);
    }

    let wf_tys = ocx.assumed_wf_types_and_report_errors(param_env, defining_use_anchor)?;
    let implied_bounds = infcx.implied_bounds_tys(param_env, def_id, &wf_tys);
    let outlives_env = OutlivesEnvironment::with_bounds(param_env, implied_bounds);
    ocx.resolve_regions_and_report_errors(defining_use_anchor, &outlives_env)?;

    if let hir::OpaqueTyOrigin::FnReturn(..) | hir::OpaqueTyOrigin::AsyncFn(..) = origin {
        // HACK: this should also fall through to the hidden type check below, but the original
        // implementation had a bug where equivalent lifetimes are not identical. This caused us
        // to reject existing stable code that is otherwise completely fine. The real fix is to
        // compare the hidden types via our type equivalence/relation infra instead of doing an
        // identity check.
        let _ = infcx.take_opaque_types();
        Ok(())
    } else {
        // Check that any hidden types found during wf checking match the hidden types that `type_of` sees.
        for (mut key, mut ty) in infcx.take_opaque_types() {
            ty.hidden_type.ty = infcx.resolve_vars_if_possible(ty.hidden_type.ty);
            key = infcx.resolve_vars_if_possible(key);
            sanity_check_found_hidden_type(tcx, key, ty.hidden_type)?;
        }
        Ok(())
    }
}

fn sanity_check_found_hidden_type<'tcx>(
    tcx: TyCtxt<'tcx>,
    key: ty::OpaqueTypeKey<'tcx>,
    mut ty: ty::OpaqueHiddenType<'tcx>,
) -> Result<(), ErrorGuaranteed> {
    if ty.ty.is_ty_var() {
        // Nothing was actually constrained.
        return Ok(());
    }
    if let ty::Alias(ty::Opaque, alias) = ty.ty.kind() {
        if alias.def_id == key.def_id.to_def_id() && alias.args == key.args {
            // Nothing was actually constrained, this is an opaque usage that was
            // only discovered to be opaque after inference vars resolved.
            return Ok(());
        }
    }
    let strip_vars = |ty: Ty<'tcx>| {
        ty.fold_with(&mut BottomUpFolder {
            tcx,
            ty_op: |t| t,
            ct_op: |c| c,
            lt_op: |l| match l.kind() {
                RegionKind::ReVar(_) => tcx.lifetimes.re_erased,
                _ => l,
            },
        })
    };
    // Closures frequently end up containing erased lifetimes in their final representation.
    // These correspond to lifetime variables that never got resolved, so we patch this up here.
    ty.ty = strip_vars(ty.ty);
    // Get the hidden type.
    let hidden_ty = tcx.type_of(key.def_id).instantiate(tcx, key.args);
    let hidden_ty = strip_vars(hidden_ty);

    // If the hidden types differ, emit a type mismatch diagnostic.
    if hidden_ty == ty.ty {
        Ok(())
    } else {
        let span = tcx.def_span(key.def_id);
        let other = ty::OpaqueHiddenType { ty: hidden_ty, span };
        Err(ty.build_mismatch_error(&other, key.def_id, tcx)?.emit())
    }
}

fn is_enum_of_nonnullable_ptr<'tcx>(
    tcx: TyCtxt<'tcx>,
    adt_def: AdtDef<'tcx>,
    args: GenericArgsRef<'tcx>,
) -> bool {
    if adt_def.repr().inhibit_enum_layout_opt() {
        return false;
    }

    let [var_one, var_two] = &adt_def.variants().raw[..] else {
        return false;
    };
    let (([], [field]) | ([field], [])) = (&var_one.fields.raw[..], &var_two.fields.raw[..]) else {
        return false;
    };
    matches!(field.ty(tcx, args).kind(), ty::FnPtr(..) | ty::Ref(..))
}

fn check_static_linkage(tcx: TyCtxt<'_>, def_id: LocalDefId) {
    if tcx.codegen_fn_attrs(def_id).import_linkage.is_some() {
        if match tcx.type_of(def_id).instantiate_identity().kind() {
            ty::RawPtr(_) => false,
            ty::Adt(adt_def, args) => !is_enum_of_nonnullable_ptr(tcx, *adt_def, *args),
            _ => true,
        } {
            tcx.dcx().emit_err(LinkageType { span: tcx.def_span(def_id) });
        }
    }
}

pub(crate) fn check_item_type(tcx: TyCtxt<'_>, def_id: LocalDefId) {
    let _indenter = indenter();
    match tcx.def_kind(def_id) {
        DefKind::Static { .. } => {
            tcx.ensure().typeck(def_id);
            maybe_check_static_with_link_section(tcx, def_id);
            check_static_inhabited(tcx, def_id);
            check_static_linkage(tcx, def_id);
        }
        DefKind::Const => {
            tcx.ensure().typeck(def_id);
        }
        DefKind::Enum => {
            check_enum(tcx, def_id);
        }
        DefKind::Fn => {
            if let Some(i) = tcx.intrinsic(def_id) {
                intrinsic::check_intrinsic_type(
                    tcx,
                    def_id,
                    tcx.def_ident_span(def_id).unwrap(),
                    i.name,
                    Abi::Rust,
                )
            }
            // Everything else is checked entirely within check_item_body
        }
        DefKind::Impl { of_trait } => {
            if of_trait && let Some(impl_trait_header) = tcx.impl_trait_header(def_id) {
                check_impl_items_against_trait(tcx, def_id, impl_trait_header);
                check_on_unimplemented(tcx, def_id);
            }
        }
        DefKind::Trait => {
            let assoc_items = tcx.associated_items(def_id);
            check_on_unimplemented(tcx, def_id);

            for &assoc_item in assoc_items.in_definition_order() {
                match assoc_item.kind {
                    ty::AssocKind::Fn => {
                        let abi = tcx.fn_sig(assoc_item.def_id).skip_binder().abi();
                        forbid_intrinsic_abi(tcx, assoc_item.ident(tcx).span, abi);
                    }
                    ty::AssocKind::Type if assoc_item.defaultness(tcx).has_value() => {
                        let trait_args = GenericArgs::identity_for_item(tcx, def_id);
                        let _: Result<_, rustc_errors::ErrorGuaranteed> = check_type_bounds(
                            tcx,
                            assoc_item,
                            assoc_item,
                            ty::TraitRef::new(tcx, def_id.to_def_id(), trait_args),
                        );
                    }
                    _ => {}
                }
            }
        }
        DefKind::Struct => {
            check_struct(tcx, def_id);
        }
        DefKind::Union => {
            check_union(tcx, def_id);
        }
        DefKind::OpaqueTy => {
            let origin = tcx.opaque_type_origin(def_id);
            if let hir::OpaqueTyOrigin::FnReturn(fn_def_id)
            | hir::OpaqueTyOrigin::AsyncFn(fn_def_id) = origin
                && let hir::Node::TraitItem(trait_item) = tcx.hir_node_by_def_id(fn_def_id)
                && let (_, hir::TraitFn::Required(..)) = trait_item.expect_fn()
            {
                // Skip opaques from RPIT in traits with no default body.
            } else {
                check_opaque(tcx, def_id);
            }
        }
        DefKind::TyAlias => {
            check_type_alias_type_params_are_used(tcx, def_id);
        }
        DefKind::ForeignMod => {
            let it = tcx.hir().expect_item(def_id);
            let hir::ItemKind::ForeignMod { abi, items } = it.kind else {
                return;
            };
            check_abi(tcx, it.hir_id(), it.span, abi);

            match abi {
                Abi::RustIntrinsic => {
                    for item in items {
                        intrinsic::check_intrinsic_type(
                            tcx,
                            item.id.owner_id.def_id,
                            item.span,
                            item.ident.name,
                            abi,
                        );
                    }
                }

                _ => {
                    for item in items {
                        let def_id = item.id.owner_id.def_id;
                        let generics = tcx.generics_of(def_id);
                        let own_counts = generics.own_counts();
                        if generics.params.len() - own_counts.lifetimes != 0 {
                            let (kinds, kinds_pl, egs) = match (own_counts.types, own_counts.consts)
                            {
                                (_, 0) => ("type", "types", Some("u32")),
                                // We don't specify an example value, because we can't generate
                                // a valid value for any type.
                                (0, _) => ("const", "consts", None),
                                _ => ("type or const", "types or consts", None),
                            };
                            struct_span_code_err!(
                                tcx.dcx(),
                                item.span,
                                E0044,
                                "foreign items may not have {kinds} parameters",
                            )
                            .with_span_label(item.span, format!("can't have {kinds} parameters"))
                            .with_help(
                                // FIXME: once we start storing spans for type arguments, turn this
                                // into a suggestion.
                                format!(
                                    "replace the {} parameters with concrete {}{}",
                                    kinds,
                                    kinds_pl,
                                    egs.map(|egs| format!(" like `{egs}`")).unwrap_or_default(),
                                ),
                            )
                            .emit();
                        }

                        let item = tcx.hir().foreign_item(item.id);
                        match &item.kind {
                            hir::ForeignItemKind::Fn(fn_decl, _, _) => {
                                require_c_abi_if_c_variadic(tcx, fn_decl, abi, item.span);
                            }
                            hir::ForeignItemKind::Static(..) => {
                                check_static_inhabited(tcx, def_id);
                                check_static_linkage(tcx, def_id);
                            }
                            _ => {}
                        }
                    }
                }
            }
        }
        DefKind::GlobalAsm => {
            let it = tcx.hir().expect_item(def_id);
            let hir::ItemKind::GlobalAsm(asm) = it.kind else {
                span_bug!(it.span, "DefKind::GlobalAsm but got {:#?}", it)
            };
            InlineAsmCtxt::new_global_asm(tcx).check_asm(asm, def_id);
        }
        _ => {}
    }
}

pub(super) fn check_on_unimplemented(tcx: TyCtxt<'_>, def_id: LocalDefId) {
    // an error would be reported if this fails.
    let _ = OnUnimplementedDirective::of_item(tcx, def_id.to_def_id());
}

pub(super) fn check_specialization_validity<'tcx>(
    tcx: TyCtxt<'tcx>,
    trait_def: &ty::TraitDef,
    trait_item: ty::AssocItem,
    impl_id: DefId,
    impl_item: DefId,
) {
    let Ok(ancestors) = trait_def.ancestors(tcx, impl_id) else { return };
    let mut ancestor_impls = ancestors.skip(1).filter_map(|parent| {
        if parent.is_from_trait() {
            None
        } else {
            Some((parent, parent.item(tcx, trait_item.def_id)))
        }
    });

    let opt_result = ancestor_impls.find_map(|(parent_impl, parent_item)| {
        match parent_item {
            // Parent impl exists, and contains the parent item we're trying to specialize, but
            // doesn't mark it `default`.
            Some(parent_item) if traits::impl_item_is_final(tcx, &parent_item) => {
                Some(Err(parent_impl.def_id()))
            }

            // Parent impl contains item and makes it specializable.
            Some(_) => Some(Ok(())),

            // Parent impl doesn't mention the item. This means it's inherited from the
            // grandparent. In that case, if parent is a `default impl`, inherited items use the
            // "defaultness" from the grandparent, else they are final.
            None => {
                if tcx.defaultness(parent_impl.def_id()).is_default() {
                    None
                } else {
                    Some(Err(parent_impl.def_id()))
                }
            }
        }
    });

    // If `opt_result` is `None`, we have only encountered `default impl`s that don't contain the
    // item. This is allowed, the item isn't actually getting specialized here.
    let result = opt_result.unwrap_or(Ok(()));

    if let Err(parent_impl) = result {
        if !tcx.is_impl_trait_in_trait(impl_item) {
            report_forbidden_specialization(tcx, impl_item, parent_impl);
        } else {
            tcx.dcx().delayed_bug(format!("parent item: {parent_impl:?} not marked as default"));
        }
    }
}

fn check_impl_items_against_trait<'tcx>(
    tcx: TyCtxt<'tcx>,
    impl_id: LocalDefId,
    impl_trait_header: ty::ImplTraitHeader<'tcx>,
) {
    let trait_ref = impl_trait_header.trait_ref.instantiate_identity();
    // If the trait reference itself is erroneous (so the compilation is going
    // to fail), skip checking the items here -- the `impl_item` table in `tcx`
    // isn't populated for such impls.
    if trait_ref.references_error() {
        return;
    }

    let impl_item_refs = tcx.associated_item_def_ids(impl_id);

    // Negative impls are not expected to have any items
    match impl_trait_header.polarity {
        ty::ImplPolarity::Reservation | ty::ImplPolarity::Positive => {}
        ty::ImplPolarity::Negative => {
            if let [first_item_ref, ..] = impl_item_refs {
                let first_item_span = tcx.def_span(first_item_ref);
                struct_span_code_err!(
                    tcx.dcx(),
                    first_item_span,
                    E0749,
                    "negative impls cannot have any items"
                )
                .emit();
            }
            return;
        }
    }

    let trait_def = tcx.trait_def(trait_ref.def_id);

    for &impl_item in impl_item_refs {
        let ty_impl_item = tcx.associated_item(impl_item);
        let ty_trait_item = if let Some(trait_item_id) = ty_impl_item.trait_item_def_id {
            tcx.associated_item(trait_item_id)
        } else {
            // Checked in `associated_item`.
            tcx.dcx().span_delayed_bug(tcx.def_span(impl_item), "missing associated item in trait");
            continue;
        };
        match ty_impl_item.kind {
            ty::AssocKind::Const => {
                tcx.ensure().compare_impl_const((
                    impl_item.expect_local(),
                    ty_impl_item.trait_item_def_id.unwrap(),
                ));
            }
            ty::AssocKind::Fn => {
                compare_impl_method(tcx, ty_impl_item, ty_trait_item, trait_ref);
            }
            ty::AssocKind::Type => {
                compare_impl_ty(tcx, ty_impl_item, ty_trait_item, trait_ref);
            }
        }

        check_specialization_validity(
            tcx,
            trait_def,
            ty_trait_item,
            impl_id.to_def_id(),
            impl_item,
        );
    }

    if let Ok(ancestors) = trait_def.ancestors(tcx, impl_id.to_def_id()) {
        // Check for missing items from trait
        let mut missing_items = Vec::new();

        let mut must_implement_one_of: Option<&[Ident]> =
            trait_def.must_implement_one_of.as_deref();

        for &trait_item_id in tcx.associated_item_def_ids(trait_ref.def_id) {
            let leaf_def = ancestors.leaf_def(tcx, trait_item_id);

            let is_implemented = leaf_def
                .as_ref()
                .is_some_and(|node_item| node_item.item.defaultness(tcx).has_value());

            if !is_implemented && tcx.defaultness(impl_id).is_final() {
                missing_items.push(tcx.associated_item(trait_item_id));
            }

            // true if this item is specifically implemented in this impl
            let is_implemented_here =
                leaf_def.as_ref().is_some_and(|node_item| !node_item.defining_node.is_from_trait());

            if !is_implemented_here {
                let full_impl_span = tcx.hir().span_with_body(tcx.local_def_id_to_hir_id(impl_id));
                match tcx.eval_default_body_stability(trait_item_id, full_impl_span) {
                    EvalResult::Deny { feature, reason, issue, .. } => default_body_is_unstable(
                        tcx,
                        full_impl_span,
                        trait_item_id,
                        feature,
                        reason,
                        issue,
                    ),

                    // Unmarked default bodies are considered stable (at least for now).
                    EvalResult::Allow | EvalResult::Unmarked => {}
                }
            }

            if let Some(required_items) = &must_implement_one_of {
                if is_implemented_here {
                    let trait_item = tcx.associated_item(trait_item_id);
                    if required_items.contains(&trait_item.ident(tcx)) {
                        must_implement_one_of = None;
                    }
                }
            }

            if let Some(leaf_def) = &leaf_def
                && !leaf_def.is_final()
                && let def_id = leaf_def.item.def_id
                && tcx.impl_method_has_trait_impl_trait_tys(def_id)
            {
                let def_kind = tcx.def_kind(def_id);
                let descr = tcx.def_kind_descr(def_kind, def_id);
                let (msg, feature) = if tcx.asyncness(def_id).is_async() {
                    (
                        format!("async {descr} in trait cannot be specialized"),
                        "async functions in traits",
                    )
                } else {
                    (
                        format!(
                            "{descr} with return-position `impl Trait` in trait cannot be specialized"
                        ),
                        "return position `impl Trait` in traits",
                    )
                };
                tcx.dcx()
                    .struct_span_err(tcx.def_span(def_id), msg)
                    .with_note(format!(
                        "specialization behaves in inconsistent and surprising ways with \
                        {feature}, and for now is disallowed"
                    ))
                    .emit();
            }
        }

        if !missing_items.is_empty() {
            let full_impl_span = tcx.hir().span_with_body(tcx.local_def_id_to_hir_id(impl_id));
            missing_items_err(tcx, impl_id, &missing_items, full_impl_span);
        }

        if let Some(missing_items) = must_implement_one_of {
            let attr_span = tcx
                .get_attr(trait_ref.def_id, sym::rustc_must_implement_one_of)
                .map(|attr| attr.span);

            missing_items_must_implement_one_of_err(
                tcx,
                tcx.def_span(impl_id),
                missing_items,
                attr_span,
            );
        }
    }
}

pub fn check_simd(tcx: TyCtxt<'_>, sp: Span, def_id: LocalDefId) {
    let t = tcx.type_of(def_id).instantiate_identity();
    if let ty::Adt(def, args) = t.kind()
        && def.is_struct()
    {
        let fields = &def.non_enum_variant().fields;
        if fields.is_empty() {
            struct_span_code_err!(tcx.dcx(), sp, E0075, "SIMD vector cannot be empty").emit();
            return;
        }
        let e = fields[FieldIdx::from_u32(0)].ty(tcx, args);
        if !fields.iter().all(|f| f.ty(tcx, args) == e) {
            struct_span_code_err!(tcx.dcx(), sp, E0076, "SIMD vector should be homogeneous")
                .with_span_label(sp, "SIMD elements must have the same type")
                .emit();
            return;
        }

        let len = if let ty::Array(_ty, c) = e.kind() {
            c.try_eval_target_usize(tcx, tcx.param_env(def.did()))
        } else {
            Some(fields.len() as u64)
        };
        if let Some(len) = len {
            if len == 0 {
                struct_span_code_err!(tcx.dcx(), sp, E0075, "SIMD vector cannot be empty").emit();
                return;
            } else if len > MAX_SIMD_LANES {
                struct_span_code_err!(
                    tcx.dcx(),
                    sp,
                    E0075,
                    "SIMD vector cannot have more than {MAX_SIMD_LANES} elements",
                )
                .emit();
                return;
            }
        }

        // Check that we use types valid for use in the lanes of a SIMD "vector register"
        // These are scalar types which directly match a "machine" type
        // Yes: Integers, floats, "thin" pointers
        // No: char, "fat" pointers, compound types
        match e.kind() {
            ty::Param(_) => (), // pass struct<T>(T, T, T, T) through, let monomorphization catch errors
            ty::Int(_) | ty::Uint(_) | ty::Float(_) | ty::RawPtr(_) => (), // struct(u8, u8, u8, u8) is ok
            ty::Array(t, _) if matches!(t.kind(), ty::Param(_)) => (), // pass struct<T>([T; N]) through, let monomorphization catch errors
            ty::Array(t, _clen)
                if matches!(t.kind(), ty::Int(_) | ty::Uint(_) | ty::Float(_) | ty::RawPtr(_)) =>
            { /* struct([f32; 4]) is ok */ }
            _ => {
                struct_span_code_err!(
                    tcx.dcx(),
                    sp,
                    E0077,
                    "SIMD vector element type should be a \
                        primitive scalar (integer/float/pointer) type"
                )
                .emit();
                return;
            }
        }
    }
}

pub(super) fn check_packed(tcx: TyCtxt<'_>, sp: Span, def: ty::AdtDef<'_>) {
    let repr = def.repr();
    if repr.packed() {
        for attr in tcx.get_attrs(def.did(), sym::repr) {
            for r in attr::parse_repr_attr(tcx.sess, attr) {
                if let attr::ReprPacked(pack) = r
                    && let Some(repr_pack) = repr.pack
                    && pack as u64 != repr_pack.bytes()
                {
                    struct_span_code_err!(
                        tcx.dcx(),
                        sp,
                        E0634,
                        "type has conflicting packed representation hints"
                    )
                    .emit();
                }
            }
        }
        if repr.align.is_some() {
            struct_span_code_err!(
                tcx.dcx(),
                sp,
                E0587,
                "type has conflicting packed and align representation hints"
            )
            .emit();
        } else {
            if let Some(def_spans) = check_packed_inner(tcx, def.did(), &mut vec![]) {
                let mut err = struct_span_code_err!(
                    tcx.dcx(),
                    sp,
                    E0588,
                    "packed type cannot transitively contain a `#[repr(align)]` type"
                );

                err.span_note(
                    tcx.def_span(def_spans[0].0),
                    format!("`{}` has a `#[repr(align)]` attribute", tcx.item_name(def_spans[0].0)),
                );

                if def_spans.len() > 2 {
                    let mut first = true;
                    for (adt_def, span) in def_spans.iter().skip(1).rev() {
                        let ident = tcx.item_name(*adt_def);
                        err.span_note(
                            *span,
                            if first {
                                format!(
                                    "`{}` contains a field of type `{}`",
                                    tcx.type_of(def.did()).instantiate_identity(),
                                    ident
                                )
                            } else {
                                format!("...which contains a field of type `{ident}`")
                            },
                        );
                        first = false;
                    }
                }

                err.emit();
            }
        }
    }
}

pub(super) fn check_packed_inner(
    tcx: TyCtxt<'_>,
    def_id: DefId,
    stack: &mut Vec<DefId>,
) -> Option<Vec<(DefId, Span)>> {
    if let ty::Adt(def, args) = tcx.type_of(def_id).instantiate_identity().kind() {
        if def.is_struct() || def.is_union() {
            if def.repr().align.is_some() {
                return Some(vec![(def.did(), DUMMY_SP)]);
            }

            stack.push(def_id);
            for field in &def.non_enum_variant().fields {
                if let ty::Adt(def, _) = field.ty(tcx, args).kind()
                    && !stack.contains(&def.did())
                    && let Some(mut defs) = check_packed_inner(tcx, def.did(), stack)
                {
                    defs.push((def.did(), field.ident(tcx).span));
                    return Some(defs);
                }
            }
            stack.pop();
        }
    }

    None
}

pub(super) fn check_transparent<'tcx>(tcx: TyCtxt<'tcx>, adt: ty::AdtDef<'tcx>) {
    if !adt.repr().transparent() {
        return;
    }

    if adt.is_union() && !tcx.features().transparent_unions {
        feature_err(
            &tcx.sess,
            sym::transparent_unions,
            tcx.def_span(adt.did()),
            "transparent unions are unstable",
        )
        .emit();
    }

    if adt.variants().len() != 1 {
        bad_variant_count(tcx, adt, tcx.def_span(adt.did()), adt.did());
        // Don't bother checking the fields.
        return;
    }

    // For each field, figure out if it's known to have "trivial" layout (i.e., is a 1-ZST), with
    // "known" respecting #[non_exhaustive] attributes.
    let field_infos = adt.all_fields().map(|field| {
        let ty = field.ty(tcx, GenericArgs::identity_for_item(tcx, field.did));
        let param_env = tcx.param_env(field.did);
        let layout = tcx.layout_of(param_env.and(ty));
        // We are currently checking the type this field came from, so it must be local
        let span = tcx.hir().span_if_local(field.did).unwrap();
        let trivial = layout.is_ok_and(|layout| layout.is_1zst());
        if !trivial {
            return (span, trivial, None);
        }
        // Even some 1-ZST fields are not allowed though, if they have `non_exhaustive`.

        fn check_non_exhaustive<'tcx>(
            tcx: TyCtxt<'tcx>,
            t: Ty<'tcx>,
        ) -> ControlFlow<(&'static str, DefId, GenericArgsRef<'tcx>, bool)> {
            match t.kind() {
                ty::Tuple(list) => list.iter().try_for_each(|t| check_non_exhaustive(tcx, t)),
                ty::Array(ty, _) => check_non_exhaustive(tcx, *ty),
                ty::Adt(def, args) => {
                    if !def.did().is_local() {
                        let non_exhaustive = def.is_variant_list_non_exhaustive()
                            || def
                                .variants()
                                .iter()
                                .any(ty::VariantDef::is_field_list_non_exhaustive);
                        let has_priv = def.all_fields().any(|f| !f.vis.is_public());
                        if non_exhaustive || has_priv {
                            return ControlFlow::Break((
                                def.descr(),
                                def.did(),
                                args,
                                non_exhaustive,
                            ));
                        }
                    }
                    def.all_fields()
                        .map(|field| field.ty(tcx, args))
                        .try_for_each(|t| check_non_exhaustive(tcx, t))
                }
                _ => ControlFlow::Continue(()),
            }
        }

        (span, trivial, check_non_exhaustive(tcx, ty).break_value())
    });

    let non_trivial_fields = field_infos
        .clone()
        .filter_map(|(span, trivial, _non_exhaustive)| if !trivial { Some(span) } else { None });
    let non_trivial_count = non_trivial_fields.clone().count();
    if non_trivial_count >= 2 {
        bad_non_zero_sized_fields(
            tcx,
            adt,
            non_trivial_count,
            non_trivial_fields,
            tcx.def_span(adt.did()),
        );
        return;
    }
    let mut prev_non_exhaustive_1zst = false;
    for (span, _trivial, non_exhaustive_1zst) in field_infos {
        if let Some((descr, def_id, args, non_exhaustive)) = non_exhaustive_1zst {
            // If there are any non-trivial fields, then there can be no non-exhaustive 1-zsts.
            // Otherwise, it's only an issue if there's >1 non-exhaustive 1-zst.
            if non_trivial_count > 0 || prev_non_exhaustive_1zst {
                tcx.node_span_lint(
                    REPR_TRANSPARENT_EXTERNAL_PRIVATE_FIELDS,
                    tcx.local_def_id_to_hir_id(adt.did().expect_local()),
                    span,
                    "zero-sized fields in `repr(transparent)` cannot \
                    contain external non-exhaustive types",
                    |lint| {
                        let note = if non_exhaustive {
                            "is marked with `#[non_exhaustive]`"
                        } else {
                            "contains private fields"
                        };
                        let field_ty = tcx.def_path_str_with_args(def_id, args);
                        lint.note(format!(
                            "this {descr} contains `{field_ty}`, which {note}, \
                                and makes it not a breaking change to become \
                                non-zero-sized in the future."
                        ));
                    },
                )
            } else {
                prev_non_exhaustive_1zst = true;
            }
        }
    }
}

#[allow(trivial_numeric_casts)]
fn check_enum(tcx: TyCtxt<'_>, def_id: LocalDefId) {
    let def = tcx.adt_def(def_id);
    def.destructor(tcx); // force the destructor to be evaluated

    if def.variants().is_empty() {
        if let Some(attr) = tcx.get_attrs(def_id, sym::repr).next() {
            struct_span_code_err!(
                tcx.dcx(),
                attr.span,
                E0084,
                "unsupported representation for zero-variant enum"
            )
            .with_span_label(tcx.def_span(def_id), "zero-variant enum")
            .emit();
        }
    }

    let repr_type_ty = def.repr().discr_type().to_ty(tcx);
    if repr_type_ty == tcx.types.i128 || repr_type_ty == tcx.types.u128 {
        if !tcx.features().repr128 {
            feature_err(
                &tcx.sess,
                sym::repr128,
                tcx.def_span(def_id),
                "repr with 128-bit type is unstable",
            )
            .emit();
        }
    }

    for v in def.variants() {
        if let ty::VariantDiscr::Explicit(discr_def_id) = v.discr {
            tcx.ensure().typeck(discr_def_id.expect_local());
        }
    }

    if def.repr().int.is_none() {
        let is_unit = |var: &ty::VariantDef| matches!(var.ctor_kind(), Some(CtorKind::Const));
        let has_disr = |var: &ty::VariantDef| matches!(var.discr, ty::VariantDiscr::Explicit(_));

        let has_non_units = def.variants().iter().any(|var| !is_unit(var));
        let disr_units = def.variants().iter().any(|var| is_unit(var) && has_disr(var));
        let disr_non_unit = def.variants().iter().any(|var| !is_unit(var) && has_disr(var));

        if disr_non_unit || (disr_units && has_non_units) {
            struct_span_code_err!(
                tcx.dcx(),
                tcx.def_span(def_id),
                E0732,
                "`#[repr(inttype)]` must be specified"
            )
            .emit();
        }
    }

    detect_discriminant_duplicate(tcx, def);
    check_transparent(tcx, def);
}

/// Part of enum check. Given the discriminants of an enum, errors if two or more discriminants are equal
fn detect_discriminant_duplicate<'tcx>(tcx: TyCtxt<'tcx>, adt: ty::AdtDef<'tcx>) {
    // Helper closure to reduce duplicate code. This gets called everytime we detect a duplicate.
    // Here `idx` refers to the order of which the discriminant appears, and its index in `vs`
    let report = |dis: Discr<'tcx>, idx, err: &mut Diag<'_>| {
        let var = adt.variant(idx); // HIR for the duplicate discriminant
        let (span, display_discr) = match var.discr {
            ty::VariantDiscr::Explicit(discr_def_id) => {
                // In the case the discriminant is both a duplicate and overflowed, let the user know
                if let hir::Node::AnonConst(expr) =
                    tcx.hir_node_by_def_id(discr_def_id.expect_local())
                    && let hir::ExprKind::Lit(lit) = &tcx.hir().body(expr.body).value.kind
                    && let rustc_ast::LitKind::Int(lit_value, _int_kind) = &lit.node
                    && *lit_value != dis.val
                {
                    (tcx.def_span(discr_def_id), format!("`{dis}` (overflowed from `{lit_value}`)"))
                } else {
                    // Otherwise, format the value as-is
                    (tcx.def_span(discr_def_id), format!("`{dis}`"))
                }
            }
            // This should not happen.
            ty::VariantDiscr::Relative(0) => (tcx.def_span(var.def_id), format!("`{dis}`")),
            ty::VariantDiscr::Relative(distance_to_explicit) => {
                // At this point we know this discriminant is a duplicate, and was not explicitly
                // assigned by the user. Here we iterate backwards to fetch the HIR for the last
                // explicitly assigned discriminant, and letting the user know that this was the
                // increment startpoint, and how many steps from there leading to the duplicate
                if let Some(explicit_idx) =
                    idx.as_u32().checked_sub(distance_to_explicit).map(VariantIdx::from_u32)
                {
                    let explicit_variant = adt.variant(explicit_idx);
                    let ve_ident = var.name;
                    let ex_ident = explicit_variant.name;
                    let sp = if distance_to_explicit > 1 { "variants" } else { "variant" };

                    err.span_label(
                        tcx.def_span(explicit_variant.def_id),
                        format!(
                            "discriminant for `{ve_ident}` incremented from this startpoint \
                            (`{ex_ident}` + {distance_to_explicit} {sp} later \
                             => `{ve_ident}` = {dis})"
                        ),
                    );
                }

                (tcx.def_span(var.def_id), format!("`{dis}`"))
            }
        };

        err.span_label(span, format!("{display_discr} assigned here"));
    };

    let mut discrs = adt.discriminants(tcx).collect::<Vec<_>>();

    // Here we loop through the discriminants, comparing each discriminant to another.
    // When a duplicate is detected, we instantiate an error and point to both
    // initial and duplicate value. The duplicate discriminant is then discarded by swapping
    // it with the last element and decrementing the `vec.len` (which is why we have to evaluate
    // `discrs.len()` anew every iteration, and why this could be tricky to do in a functional
    // style as we are mutating `discrs` on the fly).
    let mut i = 0;
    while i < discrs.len() {
        let var_i_idx = discrs[i].0;
        let mut error: Option<Diag<'_, _>> = None;

        let mut o = i + 1;
        while o < discrs.len() {
            let var_o_idx = discrs[o].0;

            if discrs[i].1.val == discrs[o].1.val {
                let err = error.get_or_insert_with(|| {
                    let mut ret = struct_span_code_err!(
                        tcx.dcx(),
                        tcx.def_span(adt.did()),
                        E0081,
                        "discriminant value `{}` assigned more than once",
                        discrs[i].1,
                    );

                    report(discrs[i].1, var_i_idx, &mut ret);

                    ret
                });

                report(discrs[o].1, var_o_idx, err);

                // Safe to unwrap here, as we wouldn't reach this point if `discrs` was empty
                discrs[o] = *discrs.last().unwrap();
                discrs.pop();
            } else {
                o += 1;
            }
        }

        if let Some(e) = error {
            e.emit();
        }

        i += 1;
    }
}

fn check_type_alias_type_params_are_used<'tcx>(tcx: TyCtxt<'tcx>, def_id: LocalDefId) {
    if tcx.type_alias_is_lazy(def_id) {
        // Since we compute the variances for lazy type aliases and already reject bivariant
        // parameters as unused, we can and should skip this check for lazy type aliases.
        return;
    }

    let generics = tcx.generics_of(def_id);
    if generics.own_counts().types == 0 {
        return;
    }

    let ty = tcx.type_of(def_id).instantiate_identity();
    if ty.references_error() {
        // If there is already another error, do not emit an error for not using a type parameter.
        assert!(tcx.dcx().has_errors().is_some());
        return;
    }

    // Lazily calculated because it is only needed in case of an error.
    let bounded_params = LazyCell::new(|| {
        tcx.explicit_predicates_of(def_id)
            .predicates
            .iter()
            .filter_map(|(predicate, span)| {
                let bounded_ty = match predicate.kind().skip_binder() {
                    ty::ClauseKind::Trait(pred) => pred.trait_ref.self_ty(),
                    ty::ClauseKind::TypeOutlives(pred) => pred.0,
                    _ => return None,
                };
                if let ty::Param(param) = bounded_ty.kind() {
                    Some((param.index, span))
                } else {
                    None
                }
            })
            // FIXME: This assumes that elaborated `Sized` bounds come first (which does hold at the
            // time of writing). This is a bit fragile since we later use the span to detect elaborated
            // `Sized` bounds. If they came last for example, this would break `Trait + /*elab*/Sized`
            // since it would overwrite the span of the user-written bound. This could be fixed by
            // folding the spans with `Span::to` which requires a bit of effort I think.
            .collect::<FxIndexMap<_, _>>()
    });

    let mut params_used = BitSet::new_empty(generics.params.len());
    for leaf in ty.walk() {
        if let GenericArgKind::Type(leaf_ty) = leaf.unpack()
            && let ty::Param(param) = leaf_ty.kind()
        {
            debug!("found use of ty param {:?}", param);
            params_used.insert(param.index);
        }
    }

    for param in &generics.params {
        if !params_used.contains(param.index)
            && let ty::GenericParamDefKind::Type { .. } = param.kind
        {
            let span = tcx.def_span(param.def_id);
            let param_name = Ident::new(param.name, span);

            // The corresponding predicates are post-`Sized`-elaboration. Therefore we
            // * check for emptiness to detect lone user-written `?Sized` bounds
            // * compare the param span to the pred span to detect lone user-written `Sized` bounds
            let has_explicit_bounds = bounded_params.is_empty()
                || (*bounded_params).get(&param.index).is_some_and(|&&pred_sp| pred_sp != span);
            let const_param_help = (!has_explicit_bounds).then_some(());

            let mut diag = tcx.dcx().create_err(errors::UnusedGenericParameter {
                span,
                param_name,
                param_def_kind: tcx.def_descr(param.def_id),
                help: errors::UnusedGenericParameterHelp::TyAlias { param_name },
                const_param_help,
            });
            diag.code(E0091);
            diag.emit();
        }
    }
}

/// Emit an error for recursive opaque types.
///
/// If this is a return `impl Trait`, find the item's return expressions and point at them. For
/// direct recursion this is enough, but for indirect recursion also point at the last intermediary
/// `impl Trait`.
///
/// If all the return expressions evaluate to `!`, then we explain that the error will go away
/// after changing it. This can happen when a user uses `panic!()` or similar as a placeholder.
fn opaque_type_cycle_error(
    tcx: TyCtxt<'_>,
    opaque_def_id: LocalDefId,
    span: Span,
) -> ErrorGuaranteed {
    let mut err = struct_span_code_err!(tcx.dcx(), span, E0720, "cannot resolve opaque type");

    let mut label = false;
    if let Some((def_id, visitor)) = get_owner_return_paths(tcx, opaque_def_id) {
        let typeck_results = tcx.typeck(def_id);
        if visitor
            .returns
            .iter()
            .filter_map(|expr| typeck_results.node_type_opt(expr.hir_id))
            .all(|ty| matches!(ty.kind(), ty::Never))
        {
            let spans = visitor
                .returns
                .iter()
                .filter(|expr| typeck_results.node_type_opt(expr.hir_id).is_some())
                .map(|expr| expr.span)
                .collect::<Vec<Span>>();
            let span_len = spans.len();
            if span_len == 1 {
                err.span_label(spans[0], "this returned value is of `!` type");
            } else {
                let mut multispan: MultiSpan = spans.clone().into();
                for span in spans {
                    multispan.push_span_label(span, "this returned value is of `!` type");
                }
                err.span_note(multispan, "these returned values have a concrete \"never\" type");
            }
            err.help("this error will resolve once the item's body returns a concrete type");
        } else {
            let mut seen = FxHashSet::default();
            seen.insert(span);
            err.span_label(span, "recursive opaque type");
            label = true;
            for (sp, ty) in visitor
                .returns
                .iter()
                .filter_map(|e| typeck_results.node_type_opt(e.hir_id).map(|t| (e.span, t)))
                .filter(|(_, ty)| !matches!(ty.kind(), ty::Never))
            {
                #[derive(Default)]
                struct OpaqueTypeCollector {
                    opaques: Vec<DefId>,
                    closures: Vec<DefId>,
                }
                impl<'tcx> ty::visit::TypeVisitor<TyCtxt<'tcx>> for OpaqueTypeCollector {
                    fn visit_ty(&mut self, t: Ty<'tcx>) {
                        match *t.kind() {
                            ty::Alias(ty::Opaque, ty::AliasTy { def_id: def, .. }) => {
                                self.opaques.push(def);
                            }
                            ty::Closure(def_id, ..) | ty::Coroutine(def_id, ..) => {
                                self.closures.push(def_id);
                                t.super_visit_with(self);
                            }
                            _ => t.super_visit_with(self),
                        }
                    }
                }

                let mut visitor = OpaqueTypeCollector::default();
                ty.visit_with(&mut visitor);
                for def_id in visitor.opaques {
                    let ty_span = tcx.def_span(def_id);
                    if !seen.contains(&ty_span) {
                        let descr = if ty.is_impl_trait() { "opaque " } else { "" };
                        err.span_label(ty_span, format!("returning this {descr}type `{ty}`"));
                        seen.insert(ty_span);
                    }
                    err.span_label(sp, format!("returning here with type `{ty}`"));
                }

                for closure_def_id in visitor.closures {
                    let Some(closure_local_did) = closure_def_id.as_local() else {
                        continue;
                    };
                    let typeck_results = tcx.typeck(closure_local_did);

                    let mut label_match = |ty: Ty<'_>, span| {
                        for arg in ty.walk() {
                            if let ty::GenericArgKind::Type(ty) = arg.unpack()
                                && let ty::Alias(
                                    ty::Opaque,
                                    ty::AliasTy { def_id: captured_def_id, .. },
                                ) = *ty.kind()
                                && captured_def_id == opaque_def_id.to_def_id()
                            {
                                err.span_label(
                                    span,
                                    format!(
                                        "{} captures itself here",
                                        tcx.def_descr(closure_def_id)
                                    ),
                                );
                            }
                        }
                    };

                    // Label any closure upvars that capture the opaque
                    for capture in typeck_results.closure_min_captures_flattened(closure_local_did)
                    {
                        label_match(capture.place.ty(), capture.get_path_span(tcx));
                    }
                    // Label any coroutine locals that capture the opaque
                    if tcx.is_coroutine(closure_def_id)
                        && let Some(coroutine_layout) = tcx.mir_coroutine_witnesses(closure_def_id)
                    {
                        for interior_ty in &coroutine_layout.field_tys {
                            label_match(interior_ty.ty, interior_ty.source_info.span);
                        }
                    }
                }
            }
        }
    }
    if !label {
        err.span_label(span, "cannot resolve opaque type");
    }
    err.emit()
}

// FIXME(@lcnr): This should not be computed per coroutine, but instead once for
// each typeck root.
pub(super) fn check_coroutine_obligations(
    tcx: TyCtxt<'_>,
    def_id: LocalDefId,
) -> Result<(), ErrorGuaranteed> {
    debug_assert!(tcx.is_coroutine(def_id.to_def_id()));

    let typeck = tcx.typeck(def_id);
    let param_env = tcx.param_env(typeck.hir_owner.def_id);

    let coroutine_interior_predicates = &typeck.coroutine_interior_predicates[&def_id];
    debug!(?coroutine_interior_predicates);

    let infcx = tcx
        .infer_ctxt()
        // typeck writeback gives us predicates with their regions erased.
        // As borrowck already has checked lifetimes, we do not need to do it again.
        .ignoring_regions()
        // Bind opaque types to type checking root, as they should have been checked by borrowck,
        // but may show up in some cases, like when (root) obligations are stalled in the new solver.
        .with_opaque_type_inference(DefiningAnchor::bind(tcx, typeck.hir_owner.def_id))
        .build();

    let mut fulfillment_cx = <dyn TraitEngine<'_>>::new(&infcx);
    for (predicate, cause) in coroutine_interior_predicates {
        let obligation = Obligation::new(tcx, cause.clone(), param_env, *predicate);
        fulfillment_cx.register_predicate_obligation(&infcx, obligation);
    }

    if (tcx.features().unsized_locals || tcx.features().unsized_fn_params)
        && let Some(coroutine) = tcx.mir_coroutine_witnesses(def_id)
    {
        for field_ty in coroutine.field_tys.iter() {
            fulfillment_cx.register_bound(
                &infcx,
                param_env,
                field_ty.ty,
                tcx.require_lang_item(hir::LangItem::Sized, Some(field_ty.source_info.span)),
                ObligationCause::new(
                    field_ty.source_info.span,
                    def_id,
                    ObligationCauseCode::SizedCoroutineInterior(def_id),
                ),
            );
        }
    }

    let errors = fulfillment_cx.select_all_or_error(&infcx);
    debug!(?errors);
    if !errors.is_empty() {
        return Err(infcx.err_ctxt().report_fulfillment_errors(errors));
    }

    // Check that any hidden types found when checking these stalled coroutine obligations
    // are valid.
    for (key, ty) in infcx.take_opaque_types() {
        let hidden_type = infcx.resolve_vars_if_possible(ty.hidden_type);
        let key = infcx.resolve_vars_if_possible(key);
        sanity_check_found_hidden_type(tcx, key, hidden_type)?;
    }

    Ok(())
}
