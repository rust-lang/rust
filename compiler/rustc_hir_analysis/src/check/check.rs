use std::cell::LazyCell;
use std::ops::ControlFlow;

use rustc_abi::FieldIdx;
use rustc_attr_data_structures::AttributeKind;
use rustc_attr_data_structures::ReprAttr::ReprPacked;
use rustc_data_structures::unord::{UnordMap, UnordSet};
use rustc_errors::codes::*;
use rustc_errors::{EmissionGuarantee, MultiSpan};
use rustc_hir::def::{CtorKind, DefKind};
use rustc_hir::{LangItem, Node, intravisit};
use rustc_infer::infer::{RegionVariableOrigin, TyCtxtInferExt};
use rustc_infer::traits::{Obligation, ObligationCauseCode};
use rustc_lint_defs::builtin::{
    REPR_TRANSPARENT_EXTERNAL_PRIVATE_FIELDS, UNSUPPORTED_CALLING_CONVENTIONS,
    UNSUPPORTED_FN_PTR_CALLING_CONVENTIONS,
};
use rustc_middle::hir::nested_filter;
use rustc_middle::middle::resolve_bound_vars::ResolvedArg;
use rustc_middle::middle::stability::EvalResult;
use rustc_middle::ty::error::TypeErrorToStringExt;
use rustc_middle::ty::layout::{LayoutError, MAX_SIMD_LANES};
use rustc_middle::ty::util::Discr;
use rustc_middle::ty::{
    AdtDef, BottomUpFolder, FnSig, GenericArgKind, RegionKind, TypeFoldable, TypeSuperVisitable,
    TypeVisitable, TypeVisitableExt, fold_regions,
};
use rustc_session::lint::builtin::UNINHABITED_STATIC;
use rustc_target::spec::{AbiMap, AbiMapping};
use rustc_trait_selection::error_reporting::InferCtxtErrorExt;
use rustc_trait_selection::error_reporting::traits::on_unimplemented::OnUnimplementedDirective;
use rustc_trait_selection::traits;
use rustc_trait_selection::traits::query::evaluate_obligation::InferCtxtExt;
use tracing::{debug, instrument};
use ty::TypingMode;
use {rustc_attr_data_structures as attrs, rustc_hir as hir};

use super::compare_impl_item::check_type_bounds;
use super::*;

fn add_abi_diag_help<T: EmissionGuarantee>(abi: ExternAbi, diag: &mut Diag<'_, T>) {
    if let ExternAbi::Cdecl { unwind } = abi {
        let c_abi = ExternAbi::C { unwind };
        diag.help(format!("use `extern {c_abi}` instead",));
    } else if let ExternAbi::Stdcall { unwind } = abi {
        let c_abi = ExternAbi::C { unwind };
        let system_abi = ExternAbi::System { unwind };
        diag.help(format!(
            "if you need `extern {abi}` on win32 and `extern {c_abi}` everywhere else, \
                use `extern {system_abi}`"
        ));
    }
}

pub fn check_abi(tcx: TyCtxt<'_>, hir_id: hir::HirId, span: Span, abi: ExternAbi) {
    // FIXME: this should be checked earlier, e.g. in `rustc_ast_lowering`, to fix
    // things like #86232.

    match AbiMap::from_target(&tcx.sess.target).canonize_abi(abi, false) {
        AbiMapping::Direct(..) => (),
        AbiMapping::Invalid => {
            let mut err = struct_span_code_err!(
                tcx.dcx(),
                span,
                E0570,
                "`{abi}` is not a supported ABI for the current target",
            );
            add_abi_diag_help(abi, &mut err);
            err.emit();
        }
        AbiMapping::Deprecated(..) => {
            tcx.node_span_lint(UNSUPPORTED_CALLING_CONVENTIONS, hir_id, span, |lint| {
                lint.primary_message("use of calling convention not supported on this target");
                add_abi_diag_help(abi, lint);
            });
        }
    }
}

pub fn check_abi_fn_ptr(tcx: TyCtxt<'_>, hir_id: hir::HirId, span: Span, abi: ExternAbi) {
    // This is always an FCW, even for `AbiMapping::Invalid`, since we started linting later than
    // in `check_abi` above.
    match AbiMap::from_target(&tcx.sess.target).canonize_abi(abi, false) {
        AbiMapping::Direct(..) => (),
        // This is not a redundant match arm: these ABIs started linting after introducing
        // UNSUPPORTED_FN_PTR_CALLING_CONVENTIONS already existed and we want to
        // avoid expanding the scope of that lint so it can move to a hard error sooner.
        AbiMapping::Deprecated(..) => {
            tcx.node_span_lint(UNSUPPORTED_CALLING_CONVENTIONS, hir_id, span, |lint| {
                lint.primary_message("use of calling convention not supported on this target");
                add_abi_diag_help(abi, lint);
            });
        }
        AbiMapping::Invalid => {
            tcx.node_span_lint(UNSUPPORTED_FN_PTR_CALLING_CONVENTIONS, hir_id, span, |lint| {
                lint.primary_message(format!(
                    "the calling convention {abi} is not supported on this target"
                ));
            });
        }
    }
}

pub fn check_custom_abi(tcx: TyCtxt<'_>, def_id: LocalDefId, fn_sig: FnSig<'_>, fn_sig_span: Span) {
    if fn_sig.abi == ExternAbi::Custom {
        // Function definitions that use `extern "custom"` must be naked functions.
        if !tcx.has_attr(def_id, sym::naked) {
            tcx.dcx().emit_err(crate::errors::AbiCustomClothedFunction {
                span: fn_sig_span,
                naked_span: tcx.def_span(def_id).shrink_to_lo(),
            });
        }
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
}

fn check_union(tcx: TyCtxt<'_>, def_id: LocalDefId) {
    let def = tcx.adt_def(def_id);
    let span = tcx.def_span(def_id);
    def.destructor(tcx); // force the destructor to be evaluated
    check_transparent(tcx, def);
    check_union_fields(tcx, span, def_id);
    check_packed(tcx, span, def);
}

fn allowed_union_or_unsafe_field<'tcx>(
    tcx: TyCtxt<'tcx>,
    ty: Ty<'tcx>,
    typing_env: ty::TypingEnv<'tcx>,
    span: Span,
) -> bool {
    // HACK (not that bad of a hack don't worry): Some codegen tests don't even define proper
    // impls for `Copy`. Let's short-circuit here for this validity check, since a lot of them
    // use unions. We should eventually fix all the tests to define that lang item or use
    // minicore stubs.
    if ty.is_trivially_pure_clone_copy() {
        return true;
    }
    // If `BikeshedGuaranteedNoDrop` is not defined in a `#[no_core]` test, fall back to `Copy`.
    // This is an underapproximation of `BikeshedGuaranteedNoDrop`,
    let def_id = tcx
        .lang_items()
        .get(LangItem::BikeshedGuaranteedNoDrop)
        .unwrap_or_else(|| tcx.require_lang_item(LangItem::Copy, span));
    let Ok(ty) = tcx.try_normalize_erasing_regions(typing_env, ty) else {
        tcx.dcx().span_delayed_bug(span, "could not normalize field type");
        return true;
    };
    let (infcx, param_env) = tcx.infer_ctxt().build_with_typing_env(typing_env);
    infcx.predicate_must_hold_modulo_regions(&Obligation::new(
        tcx,
        ObligationCause::dummy_with_span(span),
        param_env,
        ty::TraitRef::new(tcx, def_id, [ty]),
    ))
}

/// Check that the fields of the `union` do not need dropping.
fn check_union_fields(tcx: TyCtxt<'_>, span: Span, item_def_id: LocalDefId) -> bool {
    let def = tcx.adt_def(item_def_id);
    assert!(def.is_union());

    let typing_env = ty::TypingEnv::non_body_analysis(tcx, item_def_id);
    let args = ty::GenericArgs::identity_for_item(tcx, item_def_id);

    for field in &def.non_enum_variant().fields {
        if !allowed_union_or_unsafe_field(tcx, field.ty(tcx, args), typing_env, span) {
            let (field_span, ty_span) = match tcx.hir_get_if_local(field.did) {
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
        }
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
    let layout = match tcx.layout_of(ty::TypingEnv::fully_monomorphized().as_query_input(ty)) {
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
    if layout.is_uninhabited() {
        tcx.node_span_lint(
            UNINHABITED_STATIC,
            tcx.local_def_id_to_hir_id(def_id),
            span,
            |lint| {
                lint.primary_message("static of uninhabited type");
                lint
                .note("uninhabited statics cannot be initialized, and any access would be an immediate error");
            },
        );
    }
}

/// Checks that an opaque type does not contain cycles and does not use `Self` or `T::Foo`
/// projections that would result in "inheriting lifetimes".
fn check_opaque(tcx: TyCtxt<'_>, def_id: LocalDefId) {
    let hir::OpaqueTy { origin, .. } = *tcx.hir_expect_opaque_ty(def_id);

    // HACK(jynelson): trying to infer the type of `impl trait` breaks documenting
    // `async-std` (and `pub async fn` in general).
    // Since rustdoc doesn't care about the concrete type behind `impl Trait`, just don't look at it!
    // See https://github.com/rust-lang/rust/issues/75100
    if tcx.sess.opts.actually_rustdoc {
        return;
    }

    if tcx.type_of(def_id).instantiate_identity().references_error() {
        return;
    }
    if check_opaque_for_cycles(tcx, def_id).is_err() {
        return;
    }

    let _ = check_opaque_meets_bounds(tcx, def_id, origin);
}

/// Checks that an opaque type does not contain cycles.
pub(super) fn check_opaque_for_cycles<'tcx>(
    tcx: TyCtxt<'tcx>,
    def_id: LocalDefId,
) -> Result<(), ErrorGuaranteed> {
    let args = GenericArgs::identity_for_item(tcx, def_id);

    // First, try to look at any opaque expansion cycles, considering coroutine fields
    // (even though these aren't necessarily true errors).
    if tcx.try_expand_impl_trait_type(def_id.to_def_id(), args).is_err() {
        let reported = opaque_type_cycle_error(tcx, def_id);
        return Err(reported);
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
    origin: hir::OpaqueTyOrigin<LocalDefId>,
) -> Result<(), ErrorGuaranteed> {
    let (span, definition_def_id) =
        if let Some((span, def_id)) = best_definition_site_of_opaque(tcx, def_id, origin) {
            (span, Some(def_id))
        } else {
            (tcx.def_span(def_id), None)
        };

    let defining_use_anchor = match origin {
        hir::OpaqueTyOrigin::FnReturn { parent, .. }
        | hir::OpaqueTyOrigin::AsyncFn { parent, .. }
        | hir::OpaqueTyOrigin::TyAlias { parent, .. } => parent,
    };
    let param_env = tcx.param_env(defining_use_anchor);

    // FIXME(#132279): Once `PostBorrowckAnalysis` is supported in the old solver, this branch should be removed.
    let infcx = tcx.infer_ctxt().build(if tcx.next_trait_solver_globally() {
        TypingMode::post_borrowck_analysis(tcx, defining_use_anchor)
    } else {
        TypingMode::analysis_in_body(tcx, defining_use_anchor)
    });
    let ocx = ObligationCtxt::new_with_diagnostics(&infcx);

    let args = match origin {
        hir::OpaqueTyOrigin::FnReturn { parent, .. }
        | hir::OpaqueTyOrigin::AsyncFn { parent, .. }
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
    let hidden_ty = fold_regions(tcx, hidden_ty, |re, _dbi| match re.kind() {
        ty::ReErased => infcx.next_region_var(RegionVariableOrigin::MiscVariable(span)),
        _ => re,
    });

    // HACK: We eagerly instantiate some bounds to report better errors for them...
    // This isn't necessary for correctness, since we register these bounds when
    // equating the opaque below, but we should clean this up in the new solver.
    for (predicate, pred_span) in
        tcx.explicit_item_bounds(def_id).iter_instantiated_copied(tcx, args)
    {
        let predicate = predicate.fold_with(&mut BottomUpFolder {
            tcx,
            ty_op: |ty| if ty == opaque_ty { hidden_ty } else { ty },
            lt_op: |lt| lt,
            ct_op: |ct| ct,
        });

        ocx.register_obligation(Obligation::new(
            tcx,
            ObligationCause::new(
                span,
                def_id,
                ObligationCauseCode::OpaqueTypeBound(pred_span, definition_def_id),
            ),
            param_env,
            predicate,
        ));
    }

    let misc_cause = ObligationCause::misc(span, def_id);
    // FIXME: We should just register the item bounds here, rather than equating.
    // FIXME(const_trait_impl): When we do that, please make sure to also register
    // the `~const` bounds.
    match ocx.eq(&misc_cause, param_env, opaque_ty, hidden_ty) {
        Ok(()) => {}
        Err(ty_err) => {
            // Some types may be left "stranded" if they can't be reached
            // from a lowered rustc_middle bound but they're mentioned in the HIR.
            // This will happen, e.g., when a nested opaque is inside of a non-
            // existent associated type, like `impl Trait<Missing = impl Trait>`.
            // See <tests/ui/impl-trait/stranded-opaque.rs>.
            let ty_err = ty_err.to_string(tcx);
            let guar = tcx.dcx().span_delayed_bug(
                span,
                format!("could not unify `{hidden_ty}` with revealed type:\n{ty_err}"),
            );
            return Err(guar);
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
    ocx.resolve_regions_and_report_errors(defining_use_anchor, param_env, wf_tys)?;

    if infcx.next_trait_solver() {
        Ok(())
    } else if let hir::OpaqueTyOrigin::FnReturn { .. } | hir::OpaqueTyOrigin::AsyncFn { .. } =
        origin
    {
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
            ty.ty = infcx.resolve_vars_if_possible(ty.ty);
            key = infcx.resolve_vars_if_possible(key);
            sanity_check_found_hidden_type(tcx, key, ty)?;
        }
        Ok(())
    }
}

fn best_definition_site_of_opaque<'tcx>(
    tcx: TyCtxt<'tcx>,
    opaque_def_id: LocalDefId,
    origin: hir::OpaqueTyOrigin<LocalDefId>,
) -> Option<(Span, LocalDefId)> {
    struct TaitConstraintLocator<'tcx> {
        opaque_def_id: LocalDefId,
        tcx: TyCtxt<'tcx>,
    }
    impl<'tcx> TaitConstraintLocator<'tcx> {
        fn check(&self, item_def_id: LocalDefId) -> ControlFlow<(Span, LocalDefId)> {
            if !self.tcx.has_typeck_results(item_def_id) {
                return ControlFlow::Continue(());
            }

            let opaque_types_defined_by = self.tcx.opaque_types_defined_by(item_def_id);
            // Don't try to check items that cannot possibly constrain the type.
            if !opaque_types_defined_by.contains(&self.opaque_def_id) {
                return ControlFlow::Continue(());
            }

            if let Some(hidden_ty) = self
                .tcx
                .mir_borrowck(item_def_id)
                .ok()
                .and_then(|opaque_types| opaque_types.0.get(&self.opaque_def_id))
            {
                ControlFlow::Break((hidden_ty.span, item_def_id))
            } else {
                ControlFlow::Continue(())
            }
        }
    }
    impl<'tcx> intravisit::Visitor<'tcx> for TaitConstraintLocator<'tcx> {
        type NestedFilter = nested_filter::All;
        type Result = ControlFlow<(Span, LocalDefId)>;
        fn maybe_tcx(&mut self) -> Self::MaybeTyCtxt {
            self.tcx
        }
        fn visit_expr(&mut self, ex: &'tcx hir::Expr<'tcx>) -> Self::Result {
            intravisit::walk_expr(self, ex)
        }
        fn visit_item(&mut self, it: &'tcx hir::Item<'tcx>) -> Self::Result {
            self.check(it.owner_id.def_id)?;
            intravisit::walk_item(self, it)
        }
        fn visit_impl_item(&mut self, it: &'tcx hir::ImplItem<'tcx>) -> Self::Result {
            self.check(it.owner_id.def_id)?;
            intravisit::walk_impl_item(self, it)
        }
        fn visit_trait_item(&mut self, it: &'tcx hir::TraitItem<'tcx>) -> Self::Result {
            self.check(it.owner_id.def_id)?;
            intravisit::walk_trait_item(self, it)
        }
        fn visit_foreign_item(&mut self, it: &'tcx hir::ForeignItem<'tcx>) -> Self::Result {
            intravisit::walk_foreign_item(self, it)
        }
    }

    let mut locator = TaitConstraintLocator { tcx, opaque_def_id };
    match origin {
        hir::OpaqueTyOrigin::FnReturn { parent, .. }
        | hir::OpaqueTyOrigin::AsyncFn { parent, .. } => locator.check(parent).break_value(),
        hir::OpaqueTyOrigin::TyAlias { parent, in_assoc_ty: true } => {
            let impl_def_id = tcx.local_parent(parent);
            for assoc in tcx.associated_items(impl_def_id).in_definition_order() {
                match assoc.kind {
                    ty::AssocKind::Const { .. } | ty::AssocKind::Fn { .. } => {
                        if let ControlFlow::Break(span) = locator.check(assoc.def_id.expect_local())
                        {
                            return Some(span);
                        }
                    }
                    ty::AssocKind::Type { .. } => {}
                }
            }

            None
        }
        hir::OpaqueTyOrigin::TyAlias { in_assoc_ty: false, .. } => {
            tcx.hir_walk_toplevel_module(&mut locator).break_value()
        }
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
        Err(ty.build_mismatch_error(&other, tcx)?.emit())
    }
}

/// Check that the opaque's precise captures list is valid (if present).
/// We check this for regular `impl Trait`s and also RPITITs, even though the latter
/// are technically GATs.
///
/// This function is responsible for:
/// 1. Checking that all type/const params are mention in the captures list.
/// 2. Checking that all lifetimes that are implicitly captured are mentioned.
/// 3. Asserting that all parameters mentioned in the captures list are invariant.
fn check_opaque_precise_captures<'tcx>(tcx: TyCtxt<'tcx>, opaque_def_id: LocalDefId) {
    let hir::OpaqueTy { bounds, .. } = *tcx.hir_node_by_def_id(opaque_def_id).expect_opaque_ty();
    let Some(precise_capturing_args) = bounds.iter().find_map(|bound| match *bound {
        hir::GenericBound::Use(bounds, ..) => Some(bounds),
        _ => None,
    }) else {
        // No precise capturing args; nothing to validate
        return;
    };

    let mut expected_captures = UnordSet::default();
    let mut shadowed_captures = UnordSet::default();
    let mut seen_params = UnordMap::default();
    let mut prev_non_lifetime_param = None;
    for arg in precise_capturing_args {
        let (hir_id, ident) = match *arg {
            hir::PreciseCapturingArg::Param(hir::PreciseCapturingNonLifetimeArg {
                hir_id,
                ident,
                ..
            }) => {
                if prev_non_lifetime_param.is_none() {
                    prev_non_lifetime_param = Some(ident);
                }
                (hir_id, ident)
            }
            hir::PreciseCapturingArg::Lifetime(&hir::Lifetime { hir_id, ident, .. }) => {
                if let Some(prev_non_lifetime_param) = prev_non_lifetime_param {
                    tcx.dcx().emit_err(errors::LifetimesMustBeFirst {
                        lifetime_span: ident.span,
                        name: ident.name,
                        other_span: prev_non_lifetime_param.span,
                    });
                }
                (hir_id, ident)
            }
        };

        let ident = ident.normalize_to_macros_2_0();
        if let Some(span) = seen_params.insert(ident, ident.span) {
            tcx.dcx().emit_err(errors::DuplicatePreciseCapture {
                name: ident.name,
                first_span: span,
                second_span: ident.span,
            });
        }

        match tcx.named_bound_var(hir_id) {
            Some(ResolvedArg::EarlyBound(def_id)) => {
                expected_captures.insert(def_id.to_def_id());

                // Make sure we allow capturing these lifetimes through `Self` and
                // `T::Assoc` projection syntax, too. These will occur when we only
                // see lifetimes are captured after hir-lowering -- this aligns with
                // the cases that were stabilized with the `impl_trait_projection`
                // feature -- see <https://github.com/rust-lang/rust/pull/115659>.
                if let DefKind::LifetimeParam = tcx.def_kind(def_id)
                    && let Some(def_id) = tcx
                        .map_opaque_lifetime_to_parent_lifetime(def_id)
                        .opt_param_def_id(tcx, tcx.parent(opaque_def_id.to_def_id()))
                {
                    shadowed_captures.insert(def_id);
                }
            }
            _ => {
                tcx.dcx()
                    .span_delayed_bug(tcx.hir_span(hir_id), "parameter should have been resolved");
            }
        }
    }

    let variances = tcx.variances_of(opaque_def_id);
    let mut def_id = Some(opaque_def_id.to_def_id());
    while let Some(generics) = def_id {
        let generics = tcx.generics_of(generics);
        def_id = generics.parent;

        for param in &generics.own_params {
            if expected_captures.contains(&param.def_id) {
                assert_eq!(
                    variances[param.index as usize],
                    ty::Invariant,
                    "precise captured param should be invariant"
                );
                continue;
            }
            // If a param is shadowed by a early-bound (duplicated) lifetime, then
            // it may or may not be captured as invariant, depending on if it shows
            // up through `Self` or `T::Assoc` syntax.
            if shadowed_captures.contains(&param.def_id) {
                continue;
            }

            match param.kind {
                ty::GenericParamDefKind::Lifetime => {
                    let use_span = tcx.def_span(param.def_id);
                    let opaque_span = tcx.def_span(opaque_def_id);
                    // Check if the lifetime param was captured but isn't named in the precise captures list.
                    if variances[param.index as usize] == ty::Invariant {
                        if let DefKind::OpaqueTy = tcx.def_kind(tcx.parent(param.def_id))
                            && let Some(def_id) = tcx
                                .map_opaque_lifetime_to_parent_lifetime(param.def_id.expect_local())
                                .opt_param_def_id(tcx, tcx.parent(opaque_def_id.to_def_id()))
                        {
                            tcx.dcx().emit_err(errors::LifetimeNotCaptured {
                                opaque_span,
                                use_span,
                                param_span: tcx.def_span(def_id),
                            });
                        } else {
                            if tcx.def_kind(tcx.parent(param.def_id)) == DefKind::Trait {
                                tcx.dcx().emit_err(errors::LifetimeImplicitlyCaptured {
                                    opaque_span,
                                    param_span: tcx.def_span(param.def_id),
                                });
                            } else {
                                // If the `use_span` is actually just the param itself, then we must
                                // have not duplicated the lifetime but captured the original.
                                // The "effective" `use_span` will be the span of the opaque itself,
                                // and the param span will be the def span of the param.
                                tcx.dcx().emit_err(errors::LifetimeNotCaptured {
                                    opaque_span,
                                    use_span: opaque_span,
                                    param_span: use_span,
                                });
                            }
                        }
                        continue;
                    }
                }
                ty::GenericParamDefKind::Type { .. } => {
                    if matches!(tcx.def_kind(param.def_id), DefKind::Trait | DefKind::TraitAlias) {
                        // FIXME(precise_capturing): Structured suggestion for this would be useful
                        tcx.dcx().emit_err(errors::SelfTyNotCaptured {
                            trait_span: tcx.def_span(param.def_id),
                            opaque_span: tcx.def_span(opaque_def_id),
                        });
                    } else {
                        // FIXME(precise_capturing): Structured suggestion for this would be useful
                        tcx.dcx().emit_err(errors::ParamNotCaptured {
                            param_span: tcx.def_span(param.def_id),
                            opaque_span: tcx.def_span(opaque_def_id),
                            kind: "type",
                        });
                    }
                }
                ty::GenericParamDefKind::Const { .. } => {
                    // FIXME(precise_capturing): Structured suggestion for this would be useful
                    tcx.dcx().emit_err(errors::ParamNotCaptured {
                        param_span: tcx.def_span(param.def_id),
                        opaque_span: tcx.def_span(opaque_def_id),
                        kind: "const",
                    });
                }
            }
        }
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
            ty::RawPtr(_, _) => false,
            ty::Adt(adt_def, args) => !is_enum_of_nonnullable_ptr(tcx, *adt_def, *args),
            _ => true,
        } {
            tcx.dcx().emit_err(errors::LinkageType { span: tcx.def_span(def_id) });
        }
    }
}

pub(crate) fn check_item_type(tcx: TyCtxt<'_>, def_id: LocalDefId) {
    let generics = tcx.generics_of(def_id);

    for param in &generics.own_params {
        match param.kind {
            ty::GenericParamDefKind::Lifetime { .. } => {}
            ty::GenericParamDefKind::Type { has_default, .. } => {
                if has_default {
                    tcx.ensure_ok().type_of(param.def_id);
                }
            }
            ty::GenericParamDefKind::Const { has_default, .. } => {
                tcx.ensure_ok().type_of(param.def_id);
                if has_default {
                    // need to store default and type of default
                    let ct = tcx.const_param_default(param.def_id).skip_binder();
                    if let ty::ConstKind::Unevaluated(uv) = ct.kind() {
                        tcx.ensure_ok().type_of(uv.def);
                    }
                }
            }
        }
    }

    match tcx.def_kind(def_id) {
        DefKind::Static { .. } => {
            check_static_inhabited(tcx, def_id);
            check_static_linkage(tcx, def_id);
        }
        DefKind::Const => {}
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
                )
            }
        }
        DefKind::Impl { of_trait } => {
            if of_trait && let Some(impl_trait_header) = tcx.impl_trait_header(def_id) {
                if tcx
                    .ensure_ok()
                    .coherent_trait(impl_trait_header.trait_ref.instantiate_identity().def_id)
                    .is_ok()
                {
                    check_impl_items_against_trait(tcx, def_id, impl_trait_header);
                }
            }
        }
        DefKind::Trait => {
            let assoc_items = tcx.associated_items(def_id);
            check_on_unimplemented(tcx, def_id);

            for &assoc_item in assoc_items.in_definition_order() {
                match assoc_item.kind {
                    ty::AssocKind::Type { .. } if assoc_item.defaultness(tcx).has_value() => {
                        let trait_args = GenericArgs::identity_for_item(tcx, def_id);
                        let _: Result<_, rustc_errors::ErrorGuaranteed> = check_type_bounds(
                            tcx,
                            assoc_item,
                            assoc_item,
                            ty::TraitRef::new_from_args(tcx, def_id.to_def_id(), trait_args),
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
            check_opaque_precise_captures(tcx, def_id);

            let origin = tcx.local_opaque_ty_origin(def_id);
            if let hir::OpaqueTyOrigin::FnReturn { parent: fn_def_id, .. }
            | hir::OpaqueTyOrigin::AsyncFn { parent: fn_def_id, .. } = origin
                && let hir::Node::TraitItem(trait_item) = tcx.hir_node_by_def_id(fn_def_id)
                && let (_, hir::TraitFn::Required(..)) = trait_item.expect_fn()
            {
                // Skip opaques from RPIT in traits with no default body.
            } else {
                check_opaque(tcx, def_id);
            }

            tcx.ensure_ok().predicates_of(def_id);
            tcx.ensure_ok().explicit_item_bounds(def_id);
            tcx.ensure_ok().explicit_item_self_bounds(def_id);
            tcx.ensure_ok().item_bounds(def_id);
            tcx.ensure_ok().item_self_bounds(def_id);
            if tcx.is_conditionally_const(def_id) {
                tcx.ensure_ok().explicit_implied_const_bounds(def_id);
                tcx.ensure_ok().const_conditions(def_id);
            }
        }
        DefKind::TyAlias => {
            check_type_alias_type_params_are_used(tcx, def_id);
        }
        DefKind::ForeignMod => {
            let it = tcx.hir_expect_item(def_id);
            let hir::ItemKind::ForeignMod { abi, items } = it.kind else {
                return;
            };
            check_abi(tcx, it.hir_id(), it.span, abi);

            for item in items {
                let def_id = item.id.owner_id.def_id;

                let generics = tcx.generics_of(def_id);
                let own_counts = generics.own_counts();
                if generics.own_params.len() - own_counts.lifetimes != 0 {
                    let (kinds, kinds_pl, egs) = match (own_counts.types, own_counts.consts) {
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

                let item = tcx.hir_foreign_item(item.id);
                match &item.kind {
                    hir::ForeignItemKind::Fn(sig, _, _) => {
                        require_c_abi_if_c_variadic(tcx, sig.decl, abi, item.span);
                    }
                    hir::ForeignItemKind::Static(..) => {
                        check_static_inhabited(tcx, def_id);
                        check_static_linkage(tcx, def_id);
                    }
                    _ => {}
                }
            }
        }
        DefKind::Closure => {
            // This is guaranteed to be called by metadata encoding,
            // we still call it in wfcheck eagerly to ensure errors in codegen
            // attrs prevent lints from spamming the output.
            tcx.ensure_ok().codegen_fn_attrs(def_id);
            // We do not call `type_of` for closures here as that
            // depends on typecheck and would therefore hide
            // any further errors in case one typeck fails.
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

    let self_is_guaranteed_unsize_self = tcx.impl_self_is_guaranteed_unsized(impl_id);

    for &impl_item in impl_item_refs {
        let ty_impl_item = tcx.associated_item(impl_item);
        let ty_trait_item = if let Some(trait_item_id) = ty_impl_item.trait_item_def_id {
            tcx.associated_item(trait_item_id)
        } else {
            // Checked in `associated_item`.
            tcx.dcx().span_delayed_bug(tcx.def_span(impl_item), "missing associated item in trait");
            continue;
        };

        let res = tcx.ensure_ok().compare_impl_item(impl_item.expect_local());

        if res.is_ok() {
            match ty_impl_item.kind {
                ty::AssocKind::Fn { .. } => {
                    compare_impl_item::refine::check_refining_return_position_impl_trait_in_trait(
                        tcx,
                        ty_impl_item,
                        ty_trait_item,
                        tcx.impl_trait_ref(ty_impl_item.container_id(tcx))
                            .unwrap()
                            .instantiate_identity(),
                    );
                }
                ty::AssocKind::Const { .. } => {}
                ty::AssocKind::Type { .. } => {}
            }
        }

        if self_is_guaranteed_unsize_self && tcx.generics_require_sized_self(ty_trait_item.def_id) {
            tcx.emit_node_span_lint(
                rustc_lint_defs::builtin::DEAD_CODE,
                tcx.local_def_id_to_hir_id(ty_impl_item.def_id.expect_local()),
                tcx.def_span(ty_impl_item.def_id),
                errors::UselessImplItem,
            )
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

            if !is_implemented
                && tcx.defaultness(impl_id).is_final()
                // unsized types don't need to implement methods that have `Self: Sized` bounds.
                && !(self_is_guaranteed_unsize_self && tcx.generics_require_sized_self(trait_item_id))
            {
                missing_items.push(tcx.associated_item(trait_item_id));
            }

            // true if this item is specifically implemented in this impl
            let is_implemented_here =
                leaf_def.as_ref().is_some_and(|node_item| !node_item.defining_node.is_from_trait());

            if !is_implemented_here {
                let full_impl_span = tcx.hir_span_with_body(tcx.local_def_id_to_hir_id(impl_id));
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
            let full_impl_span = tcx.hir_span_with_body(tcx.local_def_id_to_hir_id(impl_id));
            missing_items_err(tcx, impl_id, &missing_items, full_impl_span);
        }

        if let Some(missing_items) = must_implement_one_of {
            let attr_span = tcx
                .get_attr(trait_ref.def_id, sym::rustc_must_implement_one_of)
                .map(|attr| attr.span());

            missing_items_must_implement_one_of_err(
                tcx,
                tcx.def_span(impl_id),
                missing_items,
                attr_span,
            );
        }
    }
}

fn check_simd(tcx: TyCtxt<'_>, sp: Span, def_id: LocalDefId) {
    let t = tcx.type_of(def_id).instantiate_identity();
    if let ty::Adt(def, args) = t.kind()
        && def.is_struct()
    {
        let fields = &def.non_enum_variant().fields;
        if fields.is_empty() {
            struct_span_code_err!(tcx.dcx(), sp, E0075, "SIMD vector cannot be empty").emit();
            return;
        }

        let array_field = &fields[FieldIdx::ZERO];
        let array_ty = array_field.ty(tcx, args);
        let ty::Array(element_ty, len_const) = array_ty.kind() else {
            struct_span_code_err!(
                tcx.dcx(),
                sp,
                E0076,
                "SIMD vector's only field must be an array"
            )
            .with_span_label(tcx.def_span(array_field.did), "not an array")
            .emit();
            return;
        };

        if let Some(second_field) = fields.get(FieldIdx::ONE) {
            struct_span_code_err!(tcx.dcx(), sp, E0075, "SIMD vector cannot have multiple fields")
                .with_span_label(tcx.def_span(second_field.did), "excess field")
                .emit();
            return;
        }

        // FIXME(repr_simd): This check is nice, but perhaps unnecessary due to the fact
        // we do not expect users to implement their own `repr(simd)` types. If they could,
        // this check is easily side-steppable by hiding the const behind normalization.
        // The consequence is that the error is, in general, only observable post-mono.
        if let Some(len) = len_const.try_to_target_usize(tcx) {
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
        // No: char, "wide" pointers, compound types
        match element_ty.kind() {
            ty::Param(_) => (), // pass struct<T>([T; 4]) through, let monomorphization catch errors
            ty::Int(_) | ty::Uint(_) | ty::Float(_) | ty::RawPtr(_, _) => (), // struct([u8; 4]) is ok
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
        if let Some(reprs) =
            attrs::find_attr!(tcx.get_all_attrs(def.did()), attrs::AttributeKind::Repr(r) => r)
        {
            for (r, _) in reprs {
                if let ReprPacked(pack) = r
                    && let Some(repr_pack) = repr.pack
                    && pack != &repr_pack
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
        } else if let Some(def_spans) = check_packed_inner(tcx, def.did(), &mut vec![]) {
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

    if adt.is_union() && !tcx.features().transparent_unions() {
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
        let typing_env = ty::TypingEnv::non_body_analysis(tcx, field.did);
        let layout = tcx.layout_of(typing_env.as_query_input(ty));
        // We are currently checking the type this field came from, so it must be local
        let span = tcx.hir_span_if_local(field.did).unwrap();
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
                    if !def.did().is_local()
                        && !attrs::find_attr!(
                            tcx.get_all_attrs(def.did()),
                            AttributeKind::PubTransparent(_)
                        )
                    {
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
                    |lint| {
                        lint.primary_message(
                            "zero-sized fields in `repr(transparent)` cannot \
                             contain external non-exhaustive types",
                        );
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
        attrs::find_attr!(
            tcx.get_all_attrs(def_id),
            attrs::AttributeKind::Repr(rs) => {
                struct_span_code_err!(
                    tcx.dcx(),
                    rs.first().unwrap().1,
                    E0084,
                    "unsupported representation for zero-variant enum"
                )
                .with_span_label(tcx.def_span(def_id), "zero-variant enum")
                .emit();
            }
        );
    }

    for v in def.variants() {
        if let ty::VariantDiscr::Explicit(discr_def_id) = v.discr {
            tcx.ensure_ok().typeck(discr_def_id.expect_local());
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
                    && let hir::ExprKind::Lit(lit) = &tcx.hir_body(expr.body).value.kind
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

    let mut params_used = DenseBitSet::new_empty(generics.own_params.len());
    for leaf in ty.walk() {
        if let GenericArgKind::Type(leaf_ty) = leaf.kind()
            && let ty::Param(param) = leaf_ty.kind()
        {
            debug!("found use of ty param {:?}", param);
            params_used.insert(param.index);
        }
    }

    for param in &generics.own_params {
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
            let const_param_help = !has_explicit_bounds;

            let mut diag = tcx.dcx().create_err(errors::UnusedGenericParameter {
                span,
                param_name,
                param_def_kind: tcx.def_descr(param.def_id),
                help: errors::UnusedGenericParameterHelp::TyAlias { param_name },
                usage_spans: vec![],
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
fn opaque_type_cycle_error(tcx: TyCtxt<'_>, opaque_def_id: LocalDefId) -> ErrorGuaranteed {
    let span = tcx.def_span(opaque_def_id);
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
                impl<'tcx> ty::TypeVisitor<TyCtxt<'tcx>> for OpaqueTypeCollector {
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
                            if let ty::GenericArgKind::Type(ty) = arg.kind()
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

pub(super) fn check_coroutine_obligations(
    tcx: TyCtxt<'_>,
    def_id: LocalDefId,
) -> Result<(), ErrorGuaranteed> {
    debug_assert!(!tcx.is_typeck_child(def_id.to_def_id()));

    let typeck_results = tcx.typeck(def_id);
    let param_env = tcx.param_env(def_id);

    debug!(?typeck_results.coroutine_stalled_predicates);

    let mode = if tcx.next_trait_solver_globally() {
        // This query is conceptually between HIR typeck and
        // MIR borrowck. We use the opaque types defined by HIR
        // and ignore region constraints.
        TypingMode::borrowck(tcx, def_id)
    } else {
        TypingMode::analysis_in_body(tcx, def_id)
    };

    // Typeck writeback gives us predicates with their regions erased.
    // We only need to check the goals while ignoring lifetimes to give good
    // error message and to avoid breaking the assumption of `mir_borrowck`
    // that all obligations already hold modulo regions.
    let infcx = tcx.infer_ctxt().ignoring_regions().build(mode);

    let ocx = ObligationCtxt::new_with_diagnostics(&infcx);
    for (predicate, cause) in &typeck_results.coroutine_stalled_predicates {
        ocx.register_obligation(Obligation::new(tcx, cause.clone(), param_env, *predicate));
    }

    let errors = ocx.select_all_or_error();
    debug!(?errors);
    if !errors.is_empty() {
        return Err(infcx.err_ctxt().report_fulfillment_errors(errors));
    }

    if !tcx.next_trait_solver_globally() {
        // Check that any hidden types found when checking these stalled coroutine obligations
        // are valid.
        for (key, ty) in infcx.take_opaque_types() {
            let hidden_type = infcx.resolve_vars_if_possible(ty);
            let key = infcx.resolve_vars_if_possible(key);
            sanity_check_found_hidden_type(tcx, key, hidden_type)?;
        }
    } else {
        // We're not checking region constraints here, so we can simply drop the
        // added opaque type uses in `TypingMode::Borrowck`.
        let _ = infcx.take_opaque_types();
    }

    Ok(())
}
