use std::cell::LazyCell;
use std::ops::{ControlFlow, Deref};

use hir::intravisit::{self, Visitor};
use rustc_abi::ExternAbi;
use rustc_data_structures::fx::{FxHashSet, FxIndexMap, FxIndexSet};
use rustc_errors::codes::*;
use rustc_errors::{Applicability, ErrorGuaranteed, pluralize, struct_span_code_err};
use rustc_hir::def::{DefKind, Res};
use rustc_hir::def_id::{DefId, LocalDefId};
use rustc_hir::lang_items::LangItem;
use rustc_hir::{AmbigArg, ItemKind};
use rustc_infer::infer::outlives::env::OutlivesEnvironment;
use rustc_infer::infer::{self, InferCtxt, TyCtxtInferExt};
use rustc_lint_defs::builtin::SUPERTRAIT_ITEM_SHADOWING_DEFINITION;
use rustc_macros::LintDiagnostic;
use rustc_middle::mir::interpret::ErrorHandled;
use rustc_middle::query::Providers;
use rustc_middle::traits::solve::NoSolution;
use rustc_middle::ty::trait_def::TraitSpecializationKind;
use rustc_middle::ty::{
    self, AdtKind, GenericArgKind, GenericArgs, GenericParamDefKind, Ty, TyCtxt, TypeFlags,
    TypeFoldable, TypeSuperVisitable, TypeVisitable, TypeVisitableExt, TypeVisitor, TypingMode,
    Upcast,
};
use rustc_middle::{bug, span_bug};
use rustc_session::parse::feature_err;
use rustc_span::{DUMMY_SP, Ident, Span, sym};
use rustc_trait_selection::error_reporting::InferCtxtErrorExt;
use rustc_trait_selection::regions::{InferCtxtRegionExt, OutlivesEnvironmentBuildExt};
use rustc_trait_selection::traits::misc::{
    ConstParamTyImplementationError, type_allowed_to_implement_const_param_ty,
};
use rustc_trait_selection::traits::query::evaluate_obligation::InferCtxtExt as _;
use rustc_trait_selection::traits::{
    self, FulfillmentError, Obligation, ObligationCause, ObligationCauseCode, ObligationCtxt,
    WellFormedLoc,
};
use tracing::{debug, instrument};
use {rustc_ast as ast, rustc_hir as hir};

use crate::autoderef::Autoderef;
use crate::constrained_generic_params::{Parameter, identify_constrained_generic_params};
use crate::errors::InvalidReceiverTyHint;
use crate::{errors, fluent_generated as fluent};

pub(super) struct WfCheckingCtxt<'a, 'tcx> {
    pub(super) ocx: ObligationCtxt<'a, 'tcx, FulfillmentError<'tcx>>,
    span: Span,
    body_def_id: LocalDefId,
    param_env: ty::ParamEnv<'tcx>,
}
impl<'a, 'tcx> Deref for WfCheckingCtxt<'a, 'tcx> {
    type Target = ObligationCtxt<'a, 'tcx, FulfillmentError<'tcx>>;
    fn deref(&self) -> &Self::Target {
        &self.ocx
    }
}

impl<'tcx> WfCheckingCtxt<'_, 'tcx> {
    fn tcx(&self) -> TyCtxt<'tcx> {
        self.ocx.infcx.tcx
    }

    // Convenience function to normalize during wfcheck. This performs
    // `ObligationCtxt::normalize`, but provides a nice `ObligationCauseCode`.
    fn normalize<T>(&self, span: Span, loc: Option<WellFormedLoc>, value: T) -> T
    where
        T: TypeFoldable<TyCtxt<'tcx>>,
    {
        self.ocx.normalize(
            &ObligationCause::new(span, self.body_def_id, ObligationCauseCode::WellFormed(loc)),
            self.param_env,
            value,
        )
    }

    /// Convenience function to *deeply* normalize during wfcheck. In the old solver,
    /// this just dispatches to [`WfCheckingCtxt::normalize`], but in the new solver
    /// this calls `deeply_normalize` and reports errors if they are encountered.
    ///
    /// This function should be called in favor of `normalize` in cases where we will
    /// then check the well-formedness of the type, since we only use the normalized
    /// signature types for implied bounds when checking regions.
    // FIXME(-Znext-solver): This should be removed when we compute implied outlives
    // bounds using the unnormalized signature of the function we're checking.
    fn deeply_normalize<T>(&self, span: Span, loc: Option<WellFormedLoc>, value: T) -> T
    where
        T: TypeFoldable<TyCtxt<'tcx>>,
    {
        if self.infcx.next_trait_solver() {
            match self.ocx.deeply_normalize(
                &ObligationCause::new(span, self.body_def_id, ObligationCauseCode::WellFormed(loc)),
                self.param_env,
                value.clone(),
            ) {
                Ok(value) => value,
                Err(errors) => {
                    self.infcx.err_ctxt().report_fulfillment_errors(errors);
                    value
                }
            }
        } else {
            self.normalize(span, loc, value)
        }
    }

    fn register_wf_obligation(&self, span: Span, loc: Option<WellFormedLoc>, term: ty::Term<'tcx>) {
        let cause = traits::ObligationCause::new(
            span,
            self.body_def_id,
            ObligationCauseCode::WellFormed(loc),
        );
        self.ocx.register_obligation(Obligation::new(
            self.tcx(),
            cause,
            self.param_env,
            ty::ClauseKind::WellFormed(term),
        ));
    }
}

pub(super) fn enter_wf_checking_ctxt<'tcx, F>(
    tcx: TyCtxt<'tcx>,
    span: Span,
    body_def_id: LocalDefId,
    f: F,
) -> Result<(), ErrorGuaranteed>
where
    F: for<'a> FnOnce(&WfCheckingCtxt<'a, 'tcx>) -> Result<(), ErrorGuaranteed>,
{
    let param_env = tcx.param_env(body_def_id);
    let infcx = &tcx.infer_ctxt().build(TypingMode::non_body_analysis());
    let ocx = ObligationCtxt::new_with_diagnostics(infcx);

    let mut wfcx = WfCheckingCtxt { ocx, span, body_def_id, param_env };

    if !tcx.features().trivial_bounds() {
        wfcx.check_false_global_bounds()
    }
    f(&mut wfcx)?;

    let errors = wfcx.select_all_or_error();
    if !errors.is_empty() {
        return Err(infcx.err_ctxt().report_fulfillment_errors(errors));
    }

    let assumed_wf_types = wfcx.ocx.assumed_wf_types_and_report_errors(param_env, body_def_id)?;
    debug!(?assumed_wf_types);

    let infcx_compat = infcx.fork();

    // We specifically want to *disable* the implied bounds hack, first,
    // so we can detect when failures are due to bevy's implied bounds.
    let outlives_env = OutlivesEnvironment::new_with_implied_bounds_compat(
        &infcx,
        body_def_id,
        param_env,
        assumed_wf_types.iter().copied(),
        true,
    );

    lint_redundant_lifetimes(tcx, body_def_id, &outlives_env);

    let errors = infcx.resolve_regions_with_outlives_env(&outlives_env);
    if errors.is_empty() {
        return Ok(());
    }

    let outlives_env = OutlivesEnvironment::new_with_implied_bounds_compat(
        &infcx_compat,
        body_def_id,
        param_env,
        assumed_wf_types,
        // Don't *disable* the implied bounds hack; though this will only apply
        // the implied bounds hack if this contains `bevy_ecs`'s `ParamSet` type.
        false,
    );
    let errors_compat = infcx_compat.resolve_regions_with_outlives_env(&outlives_env);
    if errors_compat.is_empty() {
        // FIXME: Once we fix bevy, this would be the place to insert a warning
        // to upgrade bevy.
        Ok(())
    } else {
        Err(infcx_compat.err_ctxt().report_region_errors(body_def_id, &errors_compat))
    }
}

fn check_well_formed(tcx: TyCtxt<'_>, def_id: LocalDefId) -> Result<(), ErrorGuaranteed> {
    let node = tcx.hir_node_by_def_id(def_id);
    let mut res = match node {
        hir::Node::Crate(_) => bug!("check_well_formed cannot be applied to the crate root"),
        hir::Node::Item(item) => check_item(tcx, item),
        hir::Node::TraitItem(item) => check_trait_item(tcx, item),
        hir::Node::ImplItem(item) => check_impl_item(tcx, item),
        hir::Node::ForeignItem(item) => check_foreign_item(tcx, item),
        hir::Node::ConstBlock(_) | hir::Node::Expr(_) | hir::Node::OpaqueTy(_) => {
            Ok(crate::check::check::check_item_type(tcx, def_id))
        }
        _ => unreachable!("{node:?}"),
    };

    if let Some(generics) = node.generics() {
        for param in generics.params {
            res = res.and(check_param_wf(tcx, param));
        }
    }

    res
}

/// Checks that the field types (in a struct def'n) or argument types (in an enum def'n) are
/// well-formed, meaning that they do not require any constraints not declared in the struct
/// definition itself. For example, this definition would be illegal:
///
/// ```rust
/// struct StaticRef<T> { x: &'static T }
/// ```
///
/// because the type did not declare that `T: 'static`.
///
/// We do this check as a pre-pass before checking fn bodies because if these constraints are
/// not included it frequently leads to confusing errors in fn bodies. So it's better to check
/// the types first.
#[instrument(skip(tcx), level = "debug")]
fn check_item<'tcx>(tcx: TyCtxt<'tcx>, item: &'tcx hir::Item<'tcx>) -> Result<(), ErrorGuaranteed> {
    let def_id = item.owner_id.def_id;

    debug!(
        ?item.owner_id,
        item.name = ? tcx.def_path_str(def_id)
    );
    crate::collect::lower_item(tcx, item.item_id());
    crate::collect::reject_placeholder_type_signatures_in_item(tcx, item);

    let res = match item.kind {
        // Right now we check that every default trait implementation
        // has an implementation of itself. Basically, a case like:
        //
        //     impl Trait for T {}
        //
        // has a requirement of `T: Trait` which was required for default
        // method implementations. Although this could be improved now that
        // there's a better infrastructure in place for this, it's being left
        // for a follow-up work.
        //
        // Since there's such a requirement, we need to check *just* positive
        // implementations, otherwise things like:
        //
        //     impl !Send for T {}
        //
        // won't be allowed unless there's an *explicit* implementation of `Send`
        // for `T`
        hir::ItemKind::Impl(impl_) => {
            let header = tcx.impl_trait_header(def_id);
            let is_auto = header
                .is_some_and(|header| tcx.trait_is_auto(header.trait_ref.skip_binder().def_id));

            crate::impl_wf_check::check_impl_wf(tcx, def_id)?;
            let mut res = Ok(());
            if let (hir::Defaultness::Default { .. }, true) = (impl_.defaultness, is_auto) {
                let sp = impl_.of_trait.as_ref().map_or(item.span, |t| t.path.span);
                res = Err(tcx
                    .dcx()
                    .struct_span_err(sp, "impls of auto traits cannot be default")
                    .with_span_labels(impl_.defaultness_span, "default because of this")
                    .with_span_label(sp, "auto trait")
                    .emit());
            }
            // We match on both `ty::ImplPolarity` and `ast::ImplPolarity` just to get the `!` span.
            match header.map(|h| h.polarity) {
                // `None` means this is an inherent impl
                Some(ty::ImplPolarity::Positive) | None => {
                    res = res.and(check_impl(tcx, item, impl_.self_ty, &impl_.of_trait));
                }
                Some(ty::ImplPolarity::Negative) => {
                    let ast::ImplPolarity::Negative(span) = impl_.polarity else {
                        bug!("impl_polarity query disagrees with impl's polarity in HIR");
                    };
                    // FIXME(#27579): what amount of WF checking do we need for neg impls?
                    if let hir::Defaultness::Default { .. } = impl_.defaultness {
                        let mut spans = vec![span];
                        spans.extend(impl_.defaultness_span);
                        res = Err(struct_span_code_err!(
                            tcx.dcx(),
                            spans,
                            E0750,
                            "negative impls cannot be default impls"
                        )
                        .emit());
                    }
                }
                Some(ty::ImplPolarity::Reservation) => {
                    // FIXME: what amount of WF checking do we need for reservation impls?
                }
            }
            res
        }
        hir::ItemKind::Fn { ident, sig, .. } => {
            check_item_fn(tcx, def_id, ident, item.span, sig.decl)
        }
        hir::ItemKind::Static(_, _, ty, _) => {
            check_static_item(tcx, def_id, ty.span, UnsizedHandling::Forbid)
        }
        hir::ItemKind::Const(_, _, ty, _) => check_const_item(tcx, def_id, ty.span, item.span),
        hir::ItemKind::Struct(_, generics, _) => {
            let res = check_type_defn(tcx, item, false);
            check_variances_for_type_defn(tcx, item, generics);
            res
        }
        hir::ItemKind::Union(_, generics, _) => {
            let res = check_type_defn(tcx, item, true);
            check_variances_for_type_defn(tcx, item, generics);
            res
        }
        hir::ItemKind::Enum(_, generics, _) => {
            let res = check_type_defn(tcx, item, true);
            check_variances_for_type_defn(tcx, item, generics);
            res
        }
        hir::ItemKind::Trait(..) => check_trait(tcx, item),
        hir::ItemKind::TraitAlias(..) => check_trait(tcx, item),
        // `ForeignItem`s are handled separately.
        hir::ItemKind::ForeignMod { .. } => Ok(()),
        hir::ItemKind::TyAlias(_, generics, hir_ty) if tcx.type_alias_is_lazy(item.owner_id) => {
            let res = enter_wf_checking_ctxt(tcx, item.span, def_id, |wfcx| {
                let ty = tcx.type_of(def_id).instantiate_identity();
                let item_ty =
                    wfcx.deeply_normalize(hir_ty.span, Some(WellFormedLoc::Ty(def_id)), ty);
                wfcx.register_wf_obligation(
                    hir_ty.span,
                    Some(WellFormedLoc::Ty(def_id)),
                    item_ty.into(),
                );
                check_where_clauses(wfcx, item.span, def_id);
                Ok(())
            });
            check_variances_for_type_defn(tcx, item, generics);
            res
        }
        _ => Ok(()),
    };

    crate::check::check::check_item_type(tcx, def_id);

    res
}

fn check_foreign_item<'tcx>(
    tcx: TyCtxt<'tcx>,
    item: &'tcx hir::ForeignItem<'tcx>,
) -> Result<(), ErrorGuaranteed> {
    let def_id = item.owner_id.def_id;

    debug!(
        ?item.owner_id,
        item.name = ? tcx.def_path_str(def_id)
    );

    match item.kind {
        hir::ForeignItemKind::Fn(sig, ..) => {
            check_item_fn(tcx, def_id, item.ident, item.span, sig.decl)
        }
        hir::ForeignItemKind::Static(ty, ..) => {
            check_static_item(tcx, def_id, ty.span, UnsizedHandling::AllowIfForeignTail)
        }
        hir::ForeignItemKind::Type => Ok(()),
    }
}

fn check_trait_item<'tcx>(
    tcx: TyCtxt<'tcx>,
    trait_item: &'tcx hir::TraitItem<'tcx>,
) -> Result<(), ErrorGuaranteed> {
    let def_id = trait_item.owner_id.def_id;

    crate::collect::lower_trait_item(tcx, trait_item.trait_item_id());

    let (method_sig, span) = match trait_item.kind {
        hir::TraitItemKind::Fn(ref sig, _) => (Some(sig), trait_item.span),
        hir::TraitItemKind::Type(_bounds, Some(ty)) => (None, ty.span),
        _ => (None, trait_item.span),
    };

    check_dyn_incompatible_self_trait_by_name(tcx, trait_item);

    // Check that an item definition in a subtrait is shadowing a supertrait item.
    lint_item_shadowing_supertrait_item(tcx, def_id);

    let mut res = check_associated_item(tcx, def_id, span, method_sig);

    if matches!(trait_item.kind, hir::TraitItemKind::Fn(..)) {
        for &assoc_ty_def_id in tcx.associated_types_for_impl_traits_in_associated_fn(def_id) {
            res = res.and(check_associated_item(
                tcx,
                assoc_ty_def_id.expect_local(),
                tcx.def_span(assoc_ty_def_id),
                None,
            ));
        }
    }
    res
}

/// Require that the user writes where clauses on GATs for the implicit
/// outlives bounds involving trait parameters in trait functions and
/// lifetimes passed as GAT args. See `self-outlives-lint` test.
///
/// We use the following trait as an example throughout this function:
/// ```rust,ignore (this code fails due to this lint)
/// trait IntoIter {
///     type Iter<'a>: Iterator<Item = Self::Item<'a>>;
///     type Item<'a>;
///     fn into_iter<'a>(&'a self) -> Self::Iter<'a>;
/// }
/// ```
fn check_gat_where_clauses(tcx: TyCtxt<'_>, trait_def_id: LocalDefId) {
    // Associates every GAT's def_id to a list of possibly missing bounds detected by this lint.
    let mut required_bounds_by_item = FxIndexMap::default();
    let associated_items = tcx.associated_items(trait_def_id);

    // Loop over all GATs together, because if this lint suggests adding a where-clause bound
    // to one GAT, it might then require us to an additional bound on another GAT.
    // In our `IntoIter` example, we discover a missing `Self: 'a` bound on `Iter<'a>`, which
    // then in a second loop adds a `Self: 'a` bound to `Item` due to the relationship between
    // those GATs.
    loop {
        let mut should_continue = false;
        for gat_item in associated_items.in_definition_order() {
            let gat_def_id = gat_item.def_id.expect_local();
            let gat_item = tcx.associated_item(gat_def_id);
            // If this item is not an assoc ty, or has no args, then it's not a GAT
            if !gat_item.is_type() {
                continue;
            }
            let gat_generics = tcx.generics_of(gat_def_id);
            // FIXME(jackh726): we can also warn in the more general case
            if gat_generics.is_own_empty() {
                continue;
            }

            // Gather the bounds with which all other items inside of this trait constrain the GAT.
            // This is calculated by taking the intersection of the bounds that each item
            // constrains the GAT with individually.
            let mut new_required_bounds: Option<FxIndexSet<ty::Clause<'_>>> = None;
            for item in associated_items.in_definition_order() {
                let item_def_id = item.def_id.expect_local();
                // Skip our own GAT, since it does not constrain itself at all.
                if item_def_id == gat_def_id {
                    continue;
                }

                let param_env = tcx.param_env(item_def_id);

                let item_required_bounds = match tcx.associated_item(item_def_id).kind {
                    // In our example, this corresponds to `into_iter` method
                    ty::AssocKind::Fn { .. } => {
                        // For methods, we check the function signature's return type for any GATs
                        // to constrain. In the `into_iter` case, we see that the return type
                        // `Self::Iter<'a>` is a GAT we want to gather any potential missing bounds from.
                        let sig: ty::FnSig<'_> = tcx.liberate_late_bound_regions(
                            item_def_id.to_def_id(),
                            tcx.fn_sig(item_def_id).instantiate_identity(),
                        );
                        gather_gat_bounds(
                            tcx,
                            param_env,
                            item_def_id,
                            sig.inputs_and_output,
                            // We also assume that all of the function signature's parameter types
                            // are well formed.
                            &sig.inputs().iter().copied().collect(),
                            gat_def_id,
                            gat_generics,
                        )
                    }
                    // In our example, this corresponds to the `Iter` and `Item` associated types
                    ty::AssocKind::Type { .. } => {
                        // If our associated item is a GAT with missing bounds, add them to
                        // the param-env here. This allows this GAT to propagate missing bounds
                        // to other GATs.
                        let param_env = augment_param_env(
                            tcx,
                            param_env,
                            required_bounds_by_item.get(&item_def_id),
                        );
                        gather_gat_bounds(
                            tcx,
                            param_env,
                            item_def_id,
                            tcx.explicit_item_bounds(item_def_id)
                                .iter_identity_copied()
                                .collect::<Vec<_>>(),
                            &FxIndexSet::default(),
                            gat_def_id,
                            gat_generics,
                        )
                    }
                    ty::AssocKind::Const { .. } => None,
                };

                if let Some(item_required_bounds) = item_required_bounds {
                    // Take the intersection of the required bounds for this GAT, and
                    // the item_required_bounds which are the ones implied by just
                    // this item alone.
                    // This is why we use an Option<_>, since we need to distinguish
                    // the empty set of bounds from the _uninitialized_ set of bounds.
                    if let Some(new_required_bounds) = &mut new_required_bounds {
                        new_required_bounds.retain(|b| item_required_bounds.contains(b));
                    } else {
                        new_required_bounds = Some(item_required_bounds);
                    }
                }
            }

            if let Some(new_required_bounds) = new_required_bounds {
                let required_bounds = required_bounds_by_item.entry(gat_def_id).or_default();
                if new_required_bounds.into_iter().any(|p| required_bounds.insert(p)) {
                    // Iterate until our required_bounds no longer change
                    // Since they changed here, we should continue the loop
                    should_continue = true;
                }
            }
        }
        // We know that this loop will eventually halt, since we only set `should_continue` if the
        // `required_bounds` for this item grows. Since we are not creating any new region or type
        // variables, the set of all region and type bounds that we could ever insert are limited
        // by the number of unique types and regions we observe in a given item.
        if !should_continue {
            break;
        }
    }

    for (gat_def_id, required_bounds) in required_bounds_by_item {
        // Don't suggest adding `Self: 'a` to a GAT that can't be named
        if tcx.is_impl_trait_in_trait(gat_def_id.to_def_id()) {
            continue;
        }

        let gat_item_hir = tcx.hir_expect_trait_item(gat_def_id);
        debug!(?required_bounds);
        let param_env = tcx.param_env(gat_def_id);

        let unsatisfied_bounds: Vec<_> = required_bounds
            .into_iter()
            .filter(|clause| match clause.kind().skip_binder() {
                ty::ClauseKind::RegionOutlives(ty::OutlivesPredicate(a, b)) => {
                    !region_known_to_outlive(
                        tcx,
                        gat_def_id,
                        param_env,
                        &FxIndexSet::default(),
                        a,
                        b,
                    )
                }
                ty::ClauseKind::TypeOutlives(ty::OutlivesPredicate(a, b)) => {
                    !ty_known_to_outlive(tcx, gat_def_id, param_env, &FxIndexSet::default(), a, b)
                }
                _ => bug!("Unexpected ClauseKind"),
            })
            .map(|clause| clause.to_string())
            .collect();

        if !unsatisfied_bounds.is_empty() {
            let plural = pluralize!(unsatisfied_bounds.len());
            let suggestion = format!(
                "{} {}",
                gat_item_hir.generics.add_where_or_trailing_comma(),
                unsatisfied_bounds.join(", "),
            );
            let bound =
                if unsatisfied_bounds.len() > 1 { "these bounds are" } else { "this bound is" };
            tcx.dcx()
                .struct_span_err(
                    gat_item_hir.span,
                    format!("missing required bound{} on `{}`", plural, gat_item_hir.ident),
                )
                .with_span_suggestion(
                    gat_item_hir.generics.tail_span_for_predicate_suggestion(),
                    format!("add the required where clause{plural}"),
                    suggestion,
                    Applicability::MachineApplicable,
                )
                .with_note(format!(
                    "{bound} currently required to ensure that impls have maximum flexibility"
                ))
                .with_note(
                    "we are soliciting feedback, see issue #87479 \
                     <https://github.com/rust-lang/rust/issues/87479> for more information",
                )
                .emit();
        }
    }
}

/// Add a new set of predicates to the caller_bounds of an existing param_env.
fn augment_param_env<'tcx>(
    tcx: TyCtxt<'tcx>,
    param_env: ty::ParamEnv<'tcx>,
    new_predicates: Option<&FxIndexSet<ty::Clause<'tcx>>>,
) -> ty::ParamEnv<'tcx> {
    let Some(new_predicates) = new_predicates else {
        return param_env;
    };

    if new_predicates.is_empty() {
        return param_env;
    }

    let bounds = tcx.mk_clauses_from_iter(
        param_env.caller_bounds().iter().chain(new_predicates.iter().cloned()),
    );
    // FIXME(compiler-errors): Perhaps there is a case where we need to normalize this
    // i.e. traits::normalize_param_env_or_error
    ty::ParamEnv::new(bounds)
}

/// We use the following trait as an example throughout this function.
/// Specifically, let's assume that `to_check` here is the return type
/// of `into_iter`, and the GAT we are checking this for is `Iter`.
/// ```rust,ignore (this code fails due to this lint)
/// trait IntoIter {
///     type Iter<'a>: Iterator<Item = Self::Item<'a>>;
///     type Item<'a>;
///     fn into_iter<'a>(&'a self) -> Self::Iter<'a>;
/// }
/// ```
fn gather_gat_bounds<'tcx, T: TypeFoldable<TyCtxt<'tcx>>>(
    tcx: TyCtxt<'tcx>,
    param_env: ty::ParamEnv<'tcx>,
    item_def_id: LocalDefId,
    to_check: T,
    wf_tys: &FxIndexSet<Ty<'tcx>>,
    gat_def_id: LocalDefId,
    gat_generics: &'tcx ty::Generics,
) -> Option<FxIndexSet<ty::Clause<'tcx>>> {
    // The bounds we that we would require from `to_check`
    let mut bounds = FxIndexSet::default();

    let (regions, types) = GATArgsCollector::visit(gat_def_id.to_def_id(), to_check);

    // If both regions and types are empty, then this GAT isn't in the
    // set of types we are checking, and we shouldn't try to do clause analysis
    // (particularly, doing so would end up with an empty set of clauses,
    // since the current method would require none, and we take the
    // intersection of requirements of all methods)
    if types.is_empty() && regions.is_empty() {
        return None;
    }

    for (region_a, region_a_idx) in &regions {
        // Ignore `'static` lifetimes for the purpose of this lint: it's
        // because we know it outlives everything and so doesn't give meaningful
        // clues. Also ignore `ReError`, to avoid knock-down errors.
        if let ty::ReStatic | ty::ReError(_) = region_a.kind() {
            continue;
        }
        // For each region argument (e.g., `'a` in our example), check for a
        // relationship to the type arguments (e.g., `Self`). If there is an
        // outlives relationship (`Self: 'a`), then we want to ensure that is
        // reflected in a where clause on the GAT itself.
        for (ty, ty_idx) in &types {
            // In our example, requires that `Self: 'a`
            if ty_known_to_outlive(tcx, item_def_id, param_env, wf_tys, *ty, *region_a) {
                debug!(?ty_idx, ?region_a_idx);
                debug!("required clause: {ty} must outlive {region_a}");
                // Translate into the generic parameters of the GAT. In
                // our example, the type was `Self`, which will also be
                // `Self` in the GAT.
                let ty_param = gat_generics.param_at(*ty_idx, tcx);
                let ty_param = Ty::new_param(tcx, ty_param.index, ty_param.name);
                // Same for the region. In our example, 'a corresponds
                // to the 'me parameter.
                let region_param = gat_generics.param_at(*region_a_idx, tcx);
                let region_param = ty::Region::new_early_param(
                    tcx,
                    ty::EarlyParamRegion { index: region_param.index, name: region_param.name },
                );
                // The predicate we expect to see. (In our example,
                // `Self: 'me`.)
                bounds.insert(
                    ty::ClauseKind::TypeOutlives(ty::OutlivesPredicate(ty_param, region_param))
                        .upcast(tcx),
                );
            }
        }

        // For each region argument (e.g., `'a` in our example), also check for a
        // relationship to the other region arguments. If there is an outlives
        // relationship, then we want to ensure that is reflected in the where clause
        // on the GAT itself.
        for (region_b, region_b_idx) in &regions {
            // Again, skip `'static` because it outlives everything. Also, we trivially
            // know that a region outlives itself. Also ignore `ReError`, to avoid
            // knock-down errors.
            if matches!(region_b.kind(), ty::ReStatic | ty::ReError(_)) || region_a == region_b {
                continue;
            }
            if region_known_to_outlive(tcx, item_def_id, param_env, wf_tys, *region_a, *region_b) {
                debug!(?region_a_idx, ?region_b_idx);
                debug!("required clause: {region_a} must outlive {region_b}");
                // Translate into the generic parameters of the GAT.
                let region_a_param = gat_generics.param_at(*region_a_idx, tcx);
                let region_a_param = ty::Region::new_early_param(
                    tcx,
                    ty::EarlyParamRegion { index: region_a_param.index, name: region_a_param.name },
                );
                // Same for the region.
                let region_b_param = gat_generics.param_at(*region_b_idx, tcx);
                let region_b_param = ty::Region::new_early_param(
                    tcx,
                    ty::EarlyParamRegion { index: region_b_param.index, name: region_b_param.name },
                );
                // The predicate we expect to see.
                bounds.insert(
                    ty::ClauseKind::RegionOutlives(ty::OutlivesPredicate(
                        region_a_param,
                        region_b_param,
                    ))
                    .upcast(tcx),
                );
            }
        }
    }

    Some(bounds)
}

/// Given a known `param_env` and a set of well formed types, can we prove that
/// `ty` outlives `region`.
fn ty_known_to_outlive<'tcx>(
    tcx: TyCtxt<'tcx>,
    id: LocalDefId,
    param_env: ty::ParamEnv<'tcx>,
    wf_tys: &FxIndexSet<Ty<'tcx>>,
    ty: Ty<'tcx>,
    region: ty::Region<'tcx>,
) -> bool {
    test_region_obligations(tcx, id, param_env, wf_tys, |infcx| {
        infcx.register_type_outlives_constraint_inner(infer::TypeOutlivesConstraint {
            sub_region: region,
            sup_type: ty,
            origin: infer::RelateParamBound(DUMMY_SP, ty, None),
        });
    })
}

/// Given a known `param_env` and a set of well formed types, can we prove that
/// `region_a` outlives `region_b`
fn region_known_to_outlive<'tcx>(
    tcx: TyCtxt<'tcx>,
    id: LocalDefId,
    param_env: ty::ParamEnv<'tcx>,
    wf_tys: &FxIndexSet<Ty<'tcx>>,
    region_a: ty::Region<'tcx>,
    region_b: ty::Region<'tcx>,
) -> bool {
    test_region_obligations(tcx, id, param_env, wf_tys, |infcx| {
        infcx.sub_regions(infer::RelateRegionParamBound(DUMMY_SP, None), region_b, region_a);
    })
}

/// Given a known `param_env` and a set of well formed types, set up an
/// `InferCtxt`, call the passed function (to e.g. set up region constraints
/// to be tested), then resolve region and return errors
fn test_region_obligations<'tcx>(
    tcx: TyCtxt<'tcx>,
    id: LocalDefId,
    param_env: ty::ParamEnv<'tcx>,
    wf_tys: &FxIndexSet<Ty<'tcx>>,
    add_constraints: impl FnOnce(&InferCtxt<'tcx>),
) -> bool {
    // Unfortunately, we have to use a new `InferCtxt` each call, because
    // region constraints get added and solved there and we need to test each
    // call individually.
    let infcx = tcx.infer_ctxt().build(TypingMode::non_body_analysis());

    add_constraints(&infcx);

    let errors = infcx.resolve_regions(id, param_env, wf_tys.iter().copied());
    debug!(?errors, "errors");

    // If we were able to prove that the type outlives the region without
    // an error, it must be because of the implied or explicit bounds...
    errors.is_empty()
}

/// TypeVisitor that looks for uses of GATs like
/// `<P0 as Trait<P1..Pn>>::GAT<Pn..Pm>` and adds the arguments `P0..Pm` into
/// the two vectors, `regions` and `types` (depending on their kind). For each
/// parameter `Pi` also track the index `i`.
struct GATArgsCollector<'tcx> {
    gat: DefId,
    // Which region appears and which parameter index its instantiated with
    regions: FxIndexSet<(ty::Region<'tcx>, usize)>,
    // Which params appears and which parameter index its instantiated with
    types: FxIndexSet<(Ty<'tcx>, usize)>,
}

impl<'tcx> GATArgsCollector<'tcx> {
    fn visit<T: TypeFoldable<TyCtxt<'tcx>>>(
        gat: DefId,
        t: T,
    ) -> (FxIndexSet<(ty::Region<'tcx>, usize)>, FxIndexSet<(Ty<'tcx>, usize)>) {
        let mut visitor =
            GATArgsCollector { gat, regions: FxIndexSet::default(), types: FxIndexSet::default() };
        t.visit_with(&mut visitor);
        (visitor.regions, visitor.types)
    }
}

impl<'tcx> TypeVisitor<TyCtxt<'tcx>> for GATArgsCollector<'tcx> {
    fn visit_ty(&mut self, t: Ty<'tcx>) {
        match t.kind() {
            ty::Alias(ty::Projection, p) if p.def_id == self.gat => {
                for (idx, arg) in p.args.iter().enumerate() {
                    match arg.kind() {
                        GenericArgKind::Lifetime(lt) if !lt.is_bound() => {
                            self.regions.insert((lt, idx));
                        }
                        GenericArgKind::Type(t) => {
                            self.types.insert((t, idx));
                        }
                        _ => {}
                    }
                }
            }
            _ => {}
        }
        t.super_visit_with(self)
    }
}

fn could_be_self(trait_def_id: LocalDefId, ty: &hir::Ty<'_>) -> bool {
    match ty.kind {
        hir::TyKind::TraitObject([trait_ref], ..) => match trait_ref.trait_ref.path.segments {
            [s] => s.res.opt_def_id() == Some(trait_def_id.to_def_id()),
            _ => false,
        },
        _ => false,
    }
}

/// Detect when a dyn-incompatible trait is referring to itself in one of its associated items.
///
/// In such cases, suggest using `Self` instead.
fn check_dyn_incompatible_self_trait_by_name(tcx: TyCtxt<'_>, item: &hir::TraitItem<'_>) {
    let (trait_ident, trait_def_id) =
        match tcx.hir_node_by_def_id(tcx.hir_get_parent_item(item.hir_id()).def_id) {
            hir::Node::Item(item) => match item.kind {
                hir::ItemKind::Trait(_, _, ident, ..) => (ident, item.owner_id),
                _ => return,
            },
            _ => return,
        };
    let mut trait_should_be_self = vec![];
    match &item.kind {
        hir::TraitItemKind::Const(ty, _) | hir::TraitItemKind::Type(_, Some(ty))
            if could_be_self(trait_def_id.def_id, ty) =>
        {
            trait_should_be_self.push(ty.span)
        }
        hir::TraitItemKind::Fn(sig, _) => {
            for ty in sig.decl.inputs {
                if could_be_self(trait_def_id.def_id, ty) {
                    trait_should_be_self.push(ty.span);
                }
            }
            match sig.decl.output {
                hir::FnRetTy::Return(ty) if could_be_self(trait_def_id.def_id, ty) => {
                    trait_should_be_self.push(ty.span);
                }
                _ => {}
            }
        }
        _ => {}
    }
    if !trait_should_be_self.is_empty() {
        if tcx.is_dyn_compatible(trait_def_id) {
            return;
        }
        let sugg = trait_should_be_self.iter().map(|span| (*span, "Self".to_string())).collect();
        tcx.dcx()
            .struct_span_err(
                trait_should_be_self,
                "associated item referring to unboxed trait object for its own trait",
            )
            .with_span_label(trait_ident.span, "in this trait")
            .with_multipart_suggestion(
                "you might have meant to use `Self` to refer to the implementing type",
                sugg,
                Applicability::MachineApplicable,
            )
            .emit();
    }
}

fn lint_item_shadowing_supertrait_item<'tcx>(tcx: TyCtxt<'tcx>, trait_item_def_id: LocalDefId) {
    let item_name = tcx.item_name(trait_item_def_id.to_def_id());
    let trait_def_id = tcx.local_parent(trait_item_def_id);

    let shadowed: Vec<_> = traits::supertrait_def_ids(tcx, trait_def_id.to_def_id())
        .skip(1)
        .flat_map(|supertrait_def_id| {
            tcx.associated_items(supertrait_def_id).filter_by_name_unhygienic(item_name)
        })
        .collect();
    if !shadowed.is_empty() {
        let shadowee = if let [shadowed] = shadowed[..] {
            errors::SupertraitItemShadowee::Labeled {
                span: tcx.def_span(shadowed.def_id),
                supertrait: tcx.item_name(shadowed.trait_container(tcx).unwrap()),
            }
        } else {
            let (traits, spans): (Vec<_>, Vec<_>) = shadowed
                .iter()
                .map(|item| {
                    (tcx.item_name(item.trait_container(tcx).unwrap()), tcx.def_span(item.def_id))
                })
                .unzip();
            errors::SupertraitItemShadowee::Several { traits: traits.into(), spans: spans.into() }
        };

        tcx.emit_node_span_lint(
            SUPERTRAIT_ITEM_SHADOWING_DEFINITION,
            tcx.local_def_id_to_hir_id(trait_item_def_id),
            tcx.def_span(trait_item_def_id),
            errors::SupertraitItemShadowing {
                item: item_name,
                subtrait: tcx.item_name(trait_def_id.to_def_id()),
                shadowee,
            },
        );
    }
}

fn check_impl_item<'tcx>(
    tcx: TyCtxt<'tcx>,
    impl_item: &'tcx hir::ImplItem<'tcx>,
) -> Result<(), ErrorGuaranteed> {
    crate::collect::lower_impl_item(tcx, impl_item.impl_item_id());

    let (method_sig, span) = match impl_item.kind {
        hir::ImplItemKind::Fn(ref sig, _) => (Some(sig), impl_item.span),
        // Constrain binding and overflow error spans to `<Ty>` in `type foo = <Ty>`.
        hir::ImplItemKind::Type(ty) if ty.span != DUMMY_SP => (None, ty.span),
        _ => (None, impl_item.span),
    };
    check_associated_item(tcx, impl_item.owner_id.def_id, span, method_sig)
}

fn check_param_wf(tcx: TyCtxt<'_>, param: &hir::GenericParam<'_>) -> Result<(), ErrorGuaranteed> {
    match param.kind {
        // We currently only check wf of const params here.
        hir::GenericParamKind::Lifetime { .. } | hir::GenericParamKind::Type { .. } => Ok(()),

        // Const parameters are well formed if their type is structural match.
        hir::GenericParamKind::Const { ty: hir_ty, default: _, synthetic: _ } => {
            let ty = tcx.type_of(param.def_id).instantiate_identity();

            if tcx.features().unsized_const_params() {
                enter_wf_checking_ctxt(tcx, hir_ty.span, tcx.local_parent(param.def_id), |wfcx| {
                    wfcx.register_bound(
                        ObligationCause::new(
                            hir_ty.span,
                            param.def_id,
                            ObligationCauseCode::ConstParam(ty),
                        ),
                        wfcx.param_env,
                        ty,
                        tcx.require_lang_item(LangItem::UnsizedConstParamTy, hir_ty.span),
                    );
                    Ok(())
                })
            } else if tcx.features().adt_const_params() {
                enter_wf_checking_ctxt(tcx, hir_ty.span, tcx.local_parent(param.def_id), |wfcx| {
                    wfcx.register_bound(
                        ObligationCause::new(
                            hir_ty.span,
                            param.def_id,
                            ObligationCauseCode::ConstParam(ty),
                        ),
                        wfcx.param_env,
                        ty,
                        tcx.require_lang_item(LangItem::ConstParamTy, hir_ty.span),
                    );
                    Ok(())
                })
            } else {
                let mut diag = match ty.kind() {
                    ty::Bool | ty::Char | ty::Int(_) | ty::Uint(_) | ty::Error(_) => return Ok(()),
                    ty::FnPtr(..) => tcx.dcx().struct_span_err(
                        hir_ty.span,
                        "using function pointers as const generic parameters is forbidden",
                    ),
                    ty::RawPtr(_, _) => tcx.dcx().struct_span_err(
                        hir_ty.span,
                        "using raw pointers as const generic parameters is forbidden",
                    ),
                    _ => {
                        // Avoid showing "{type error}" to users. See #118179.
                        ty.error_reported()?;

                        tcx.dcx().struct_span_err(
                            hir_ty.span,
                            format!(
                                "`{ty}` is forbidden as the type of a const generic parameter",
                            ),
                        )
                    }
                };

                diag.note("the only supported types are integers, `bool`, and `char`");

                let cause = ObligationCause::misc(hir_ty.span, param.def_id);
                let adt_const_params_feature_string =
                    " more complex and user defined types".to_string();
                let may_suggest_feature = match type_allowed_to_implement_const_param_ty(
                    tcx,
                    tcx.param_env(param.def_id),
                    ty,
                    LangItem::ConstParamTy,
                    cause,
                ) {
                    // Can never implement `ConstParamTy`, don't suggest anything.
                    Err(
                        ConstParamTyImplementationError::NotAnAdtOrBuiltinAllowed
                        | ConstParamTyImplementationError::InvalidInnerTyOfBuiltinTy(..),
                    ) => None,
                    Err(ConstParamTyImplementationError::UnsizedConstParamsFeatureRequired) => {
                        Some(vec![
                            (adt_const_params_feature_string, sym::adt_const_params),
                            (
                                " references to implement the `ConstParamTy` trait".into(),
                                sym::unsized_const_params,
                            ),
                        ])
                    }
                    // May be able to implement `ConstParamTy`. Only emit the feature help
                    // if the type is local, since the user may be able to fix the local type.
                    Err(ConstParamTyImplementationError::InfrigingFields(..)) => {
                        fn ty_is_local(ty: Ty<'_>) -> bool {
                            match ty.kind() {
                                ty::Adt(adt_def, ..) => adt_def.did().is_local(),
                                // Arrays and slices use the inner type's `ConstParamTy`.
                                ty::Array(ty, ..) | ty::Slice(ty) => ty_is_local(*ty),
                                // `&` references use the inner type's `ConstParamTy`.
                                // `&mut` are not supported.
                                ty::Ref(_, ty, ast::Mutability::Not) => ty_is_local(*ty),
                                // Say that a tuple is local if any of its components are local.
                                // This is not strictly correct, but it's likely that the user can fix the local component.
                                ty::Tuple(tys) => tys.iter().any(|ty| ty_is_local(ty)),
                                _ => false,
                            }
                        }

                        ty_is_local(ty).then_some(vec![(
                            adt_const_params_feature_string,
                            sym::adt_const_params,
                        )])
                    }
                    // Implements `ConstParamTy`, suggest adding the feature to enable.
                    Ok(..) => Some(vec![(adt_const_params_feature_string, sym::adt_const_params)]),
                };
                if let Some(features) = may_suggest_feature {
                    tcx.disabled_nightly_features(&mut diag, Some(param.hir_id), features);
                }

                Err(diag.emit())
            }
        }
    }
}

#[instrument(level = "debug", skip(tcx, span, sig_if_method))]
fn check_associated_item(
    tcx: TyCtxt<'_>,
    item_id: LocalDefId,
    span: Span,
    sig_if_method: Option<&hir::FnSig<'_>>,
) -> Result<(), ErrorGuaranteed> {
    let loc = Some(WellFormedLoc::Ty(item_id));
    enter_wf_checking_ctxt(tcx, span, item_id, |wfcx| {
        let item = tcx.associated_item(item_id);

        // Avoid bogus "type annotations needed `Foo: Bar`" errors on `impl Bar for Foo` in case
        // other `Foo` impls are incoherent.
        tcx.ensure_ok()
            .coherent_trait(tcx.parent(item.trait_item_def_id.unwrap_or(item_id.into())))?;

        let self_ty = match item.container {
            ty::AssocItemContainer::Trait => tcx.types.self_param,
            ty::AssocItemContainer::Impl => {
                tcx.type_of(item.container_id(tcx)).instantiate_identity()
            }
        };

        match item.kind {
            ty::AssocKind::Const { .. } => {
                let ty = tcx.type_of(item.def_id).instantiate_identity();
                let ty = wfcx.deeply_normalize(span, Some(WellFormedLoc::Ty(item_id)), ty);
                wfcx.register_wf_obligation(span, loc, ty.into());
                check_sized_if_body(
                    wfcx,
                    item.def_id.expect_local(),
                    ty,
                    Some(span),
                    ObligationCauseCode::SizedConstOrStatic,
                );
                Ok(())
            }
            ty::AssocKind::Fn { .. } => {
                let sig = tcx.fn_sig(item.def_id).instantiate_identity();
                let hir_sig = sig_if_method.expect("bad signature for method");
                check_fn_or_method(
                    wfcx,
                    item.ident(tcx).span,
                    sig,
                    hir_sig.decl,
                    item.def_id.expect_local(),
                );
                check_method_receiver(wfcx, hir_sig, item, self_ty)
            }
            ty::AssocKind::Type { .. } => {
                if let ty::AssocItemContainer::Trait = item.container {
                    check_associated_type_bounds(wfcx, item, span)
                }
                if item.defaultness(tcx).has_value() {
                    let ty = tcx.type_of(item.def_id).instantiate_identity();
                    let ty = wfcx.deeply_normalize(span, Some(WellFormedLoc::Ty(item_id)), ty);
                    wfcx.register_wf_obligation(span, loc, ty.into());
                }
                Ok(())
            }
        }
    })
}

/// In a type definition, we check that to ensure that the types of the fields are well-formed.
fn check_type_defn<'tcx>(
    tcx: TyCtxt<'tcx>,
    item: &hir::Item<'tcx>,
    all_sized: bool,
) -> Result<(), ErrorGuaranteed> {
    let _ = tcx.representability(item.owner_id.def_id);
    let adt_def = tcx.adt_def(item.owner_id);

    enter_wf_checking_ctxt(tcx, item.span, item.owner_id.def_id, |wfcx| {
        let variants = adt_def.variants();
        let packed = adt_def.repr().packed();

        for variant in variants.iter() {
            // All field types must be well-formed.
            for field in &variant.fields {
                if let Some(def_id) = field.value
                    && let Some(_ty) = tcx.type_of(def_id).no_bound_vars()
                {
                    // FIXME(generic_const_exprs, default_field_values): this is a hack and needs to
                    // be refactored to check the instantiate-ability of the code better.
                    if let Some(def_id) = def_id.as_local()
                        && let hir::Node::AnonConst(anon) = tcx.hir_node_by_def_id(def_id)
                        && let expr = &tcx.hir_body(anon.body).value
                        && let hir::ExprKind::Path(hir::QPath::Resolved(None, path)) = expr.kind
                        && let Res::Def(DefKind::ConstParam, _def_id) = path.res
                    {
                        // Do not evaluate bare `const` params, as those would ICE and are only
                        // usable if `#![feature(generic_const_exprs)]` is enabled.
                    } else {
                        // Evaluate the constant proactively, to emit an error if the constant has
                        // an unconditional error. We only do so if the const has no type params.
                        let _ = tcx.const_eval_poly(def_id);
                    }
                }
                let field_id = field.did.expect_local();
                let hir::FieldDef { ty: hir_ty, .. } =
                    tcx.hir_node_by_def_id(field_id).expect_field();
                let ty = wfcx.deeply_normalize(
                    hir_ty.span,
                    None,
                    tcx.type_of(field.did).instantiate_identity(),
                );
                wfcx.register_wf_obligation(
                    hir_ty.span,
                    Some(WellFormedLoc::Ty(field_id)),
                    ty.into(),
                )
            }

            // For DST, or when drop needs to copy things around, all
            // intermediate types must be sized.
            let needs_drop_copy = || {
                packed && {
                    let ty = tcx.type_of(variant.tail().did).instantiate_identity();
                    let ty = tcx.erase_regions(ty);
                    assert!(!ty.has_infer());
                    ty.needs_drop(tcx, wfcx.infcx.typing_env(wfcx.param_env))
                }
            };
            // All fields (except for possibly the last) should be sized.
            let all_sized = all_sized || variant.fields.is_empty() || needs_drop_copy();
            let unsized_len = if all_sized { 0 } else { 1 };
            for (idx, field) in
                variant.fields.raw[..variant.fields.len() - unsized_len].iter().enumerate()
            {
                let last = idx == variant.fields.len() - 1;
                let field_id = field.did.expect_local();
                let hir::FieldDef { ty: hir_ty, .. } =
                    tcx.hir_node_by_def_id(field_id).expect_field();
                let ty = wfcx.normalize(
                    hir_ty.span,
                    None,
                    tcx.type_of(field.did).instantiate_identity(),
                );
                wfcx.register_bound(
                    traits::ObligationCause::new(
                        hir_ty.span,
                        wfcx.body_def_id,
                        ObligationCauseCode::FieldSized {
                            adt_kind: match &item.kind {
                                ItemKind::Struct(..) => AdtKind::Struct,
                                ItemKind::Union(..) => AdtKind::Union,
                                ItemKind::Enum(..) => AdtKind::Enum,
                                kind => span_bug!(
                                    item.span,
                                    "should be wfchecking an ADT, got {kind:?}"
                                ),
                            },
                            span: hir_ty.span,
                            last,
                        },
                    ),
                    wfcx.param_env,
                    ty,
                    tcx.require_lang_item(LangItem::Sized, hir_ty.span),
                );
            }

            // Explicit `enum` discriminant values must const-evaluate successfully.
            if let ty::VariantDiscr::Explicit(discr_def_id) = variant.discr {
                match tcx.const_eval_poly(discr_def_id) {
                    Ok(_) => {}
                    Err(ErrorHandled::Reported(..)) => {}
                    Err(ErrorHandled::TooGeneric(sp)) => {
                        span_bug!(sp, "enum variant discr was too generic to eval")
                    }
                }
            }
        }

        check_where_clauses(wfcx, item.span, item.owner_id.def_id);
        Ok(())
    })
}

#[instrument(skip(tcx, item))]
fn check_trait(tcx: TyCtxt<'_>, item: &hir::Item<'_>) -> Result<(), ErrorGuaranteed> {
    debug!(?item.owner_id);

    let def_id = item.owner_id.def_id;
    if tcx.is_lang_item(def_id.into(), LangItem::PointeeSized) {
        // `PointeeSized` is removed during lowering.
        return Ok(());
    }

    let trait_def = tcx.trait_def(def_id);
    if trait_def.is_marker
        || matches!(trait_def.specialization_kind, TraitSpecializationKind::Marker)
    {
        for associated_def_id in &*tcx.associated_item_def_ids(def_id) {
            struct_span_code_err!(
                tcx.dcx(),
                tcx.def_span(*associated_def_id),
                E0714,
                "marker traits cannot have associated items",
            )
            .emit();
        }
    }

    let res = enter_wf_checking_ctxt(tcx, item.span, def_id, |wfcx| {
        check_where_clauses(wfcx, item.span, def_id);
        Ok(())
    });

    // Only check traits, don't check trait aliases
    if let hir::ItemKind::Trait(..) = item.kind {
        check_gat_where_clauses(tcx, item.owner_id.def_id);
    }
    res
}

/// Checks all associated type defaults of trait `trait_def_id`.
///
/// Assuming the defaults are used, check that all predicates (bounds on the
/// assoc type and where clauses on the trait) hold.
fn check_associated_type_bounds(wfcx: &WfCheckingCtxt<'_, '_>, item: ty::AssocItem, span: Span) {
    let bounds = wfcx.tcx().explicit_item_bounds(item.def_id);

    debug!("check_associated_type_bounds: bounds={:?}", bounds);
    let wf_obligations = bounds.iter_identity_copied().flat_map(|(bound, bound_span)| {
        let normalized_bound = wfcx.normalize(span, None, bound);
        traits::wf::clause_obligations(
            wfcx.infcx,
            wfcx.param_env,
            wfcx.body_def_id,
            normalized_bound,
            bound_span,
        )
    });

    wfcx.register_obligations(wf_obligations);
}

fn check_item_fn(
    tcx: TyCtxt<'_>,
    def_id: LocalDefId,
    ident: Ident,
    span: Span,
    decl: &hir::FnDecl<'_>,
) -> Result<(), ErrorGuaranteed> {
    enter_wf_checking_ctxt(tcx, span, def_id, |wfcx| {
        let sig = tcx.fn_sig(def_id).instantiate_identity();
        check_fn_or_method(wfcx, ident.span, sig, decl, def_id);
        Ok(())
    })
}

enum UnsizedHandling {
    Forbid,
    AllowIfForeignTail,
}

#[instrument(level = "debug", skip(tcx, ty_span, unsized_handling))]
fn check_static_item(
    tcx: TyCtxt<'_>,
    item_id: LocalDefId,
    ty_span: Span,
    unsized_handling: UnsizedHandling,
) -> Result<(), ErrorGuaranteed> {
    enter_wf_checking_ctxt(tcx, ty_span, item_id, |wfcx| {
        let ty = tcx.type_of(item_id).instantiate_identity();
        let item_ty = wfcx.deeply_normalize(ty_span, Some(WellFormedLoc::Ty(item_id)), ty);

        let forbid_unsized = match unsized_handling {
            UnsizedHandling::Forbid => true,
            UnsizedHandling::AllowIfForeignTail => {
                let tail =
                    tcx.struct_tail_for_codegen(item_ty, wfcx.infcx.typing_env(wfcx.param_env));
                !matches!(tail.kind(), ty::Foreign(_))
            }
        };

        wfcx.register_wf_obligation(ty_span, Some(WellFormedLoc::Ty(item_id)), item_ty.into());
        if forbid_unsized {
            wfcx.register_bound(
                traits::ObligationCause::new(
                    ty_span,
                    wfcx.body_def_id,
                    ObligationCauseCode::SizedConstOrStatic,
                ),
                wfcx.param_env,
                item_ty,
                tcx.require_lang_item(LangItem::Sized, ty_span),
            );
        }

        // Ensure that the end result is `Sync` in a non-thread local `static`.
        let should_check_for_sync = tcx.static_mutability(item_id.to_def_id())
            == Some(hir::Mutability::Not)
            && !tcx.is_foreign_item(item_id.to_def_id())
            && !tcx.is_thread_local_static(item_id.to_def_id());

        if should_check_for_sync {
            wfcx.register_bound(
                traits::ObligationCause::new(
                    ty_span,
                    wfcx.body_def_id,
                    ObligationCauseCode::SharedStatic,
                ),
                wfcx.param_env,
                item_ty,
                tcx.require_lang_item(LangItem::Sync, ty_span),
            );
        }
        Ok(())
    })
}

fn check_const_item(
    tcx: TyCtxt<'_>,
    def_id: LocalDefId,
    ty_span: Span,
    item_span: Span,
) -> Result<(), ErrorGuaranteed> {
    enter_wf_checking_ctxt(tcx, ty_span, def_id, |wfcx| {
        let ty = tcx.type_of(def_id).instantiate_identity();
        let ty = wfcx.deeply_normalize(ty_span, Some(WellFormedLoc::Ty(def_id)), ty);

        wfcx.register_wf_obligation(ty_span, Some(WellFormedLoc::Ty(def_id)), ty.into());
        wfcx.register_bound(
            traits::ObligationCause::new(
                ty_span,
                wfcx.body_def_id,
                ObligationCauseCode::SizedConstOrStatic,
            ),
            wfcx.param_env,
            ty,
            tcx.require_lang_item(LangItem::Sized, ty_span),
        );

        check_where_clauses(wfcx, item_span, def_id);

        Ok(())
    })
}

#[instrument(level = "debug", skip(tcx, hir_self_ty, hir_trait_ref))]
fn check_impl<'tcx>(
    tcx: TyCtxt<'tcx>,
    item: &'tcx hir::Item<'tcx>,
    hir_self_ty: &hir::Ty<'_>,
    hir_trait_ref: &Option<hir::TraitRef<'_>>,
) -> Result<(), ErrorGuaranteed> {
    enter_wf_checking_ctxt(tcx, item.span, item.owner_id.def_id, |wfcx| {
        match hir_trait_ref {
            Some(hir_trait_ref) => {
                // `#[rustc_reservation_impl]` impls are not real impls and
                // therefore don't need to be WF (the trait's `Self: Trait` predicate
                // won't hold).
                let trait_ref = tcx.impl_trait_ref(item.owner_id).unwrap().instantiate_identity();
                // Avoid bogus "type annotations needed `Foo: Bar`" errors on `impl Bar for Foo` in case
                // other `Foo` impls are incoherent.
                tcx.ensure_ok().coherent_trait(trait_ref.def_id)?;
                let trait_span = hir_trait_ref.path.span;
                let trait_ref = wfcx.deeply_normalize(
                    trait_span,
                    Some(WellFormedLoc::Ty(item.hir_id().expect_owner().def_id)),
                    trait_ref,
                );
                let trait_pred =
                    ty::TraitPredicate { trait_ref, polarity: ty::PredicatePolarity::Positive };
                let mut obligations = traits::wf::trait_obligations(
                    wfcx.infcx,
                    wfcx.param_env,
                    wfcx.body_def_id,
                    trait_pred,
                    trait_span,
                    item,
                );
                for obligation in &mut obligations {
                    if obligation.cause.span != trait_span {
                        // We already have a better span.
                        continue;
                    }
                    if let Some(pred) = obligation.predicate.as_trait_clause()
                        && pred.skip_binder().self_ty() == trait_ref.self_ty()
                    {
                        obligation.cause.span = hir_self_ty.span;
                    }
                    if let Some(pred) = obligation.predicate.as_projection_clause()
                        && pred.skip_binder().self_ty() == trait_ref.self_ty()
                    {
                        obligation.cause.span = hir_self_ty.span;
                    }
                }

                // Ensure that the `~const` where clauses of the trait hold for the impl.
                if tcx.is_conditionally_const(item.owner_id.def_id) {
                    for (bound, _) in
                        tcx.const_conditions(trait_ref.def_id).instantiate(tcx, trait_ref.args)
                    {
                        let bound = wfcx.normalize(
                            item.span,
                            Some(WellFormedLoc::Ty(item.hir_id().expect_owner().def_id)),
                            bound,
                        );
                        wfcx.register_obligation(Obligation::new(
                            tcx,
                            ObligationCause::new(
                                hir_self_ty.span,
                                wfcx.body_def_id,
                                ObligationCauseCode::WellFormed(None),
                            ),
                            wfcx.param_env,
                            bound.to_host_effect_clause(tcx, ty::BoundConstness::Maybe),
                        ))
                    }
                }

                debug!(?obligations);
                wfcx.register_obligations(obligations);
            }
            None => {
                let self_ty = tcx.type_of(item.owner_id).instantiate_identity();
                let self_ty = wfcx.deeply_normalize(
                    item.span,
                    Some(WellFormedLoc::Ty(item.hir_id().expect_owner().def_id)),
                    self_ty,
                );
                wfcx.register_wf_obligation(
                    hir_self_ty.span,
                    Some(WellFormedLoc::Ty(item.hir_id().expect_owner().def_id)),
                    self_ty.into(),
                );
            }
        }

        check_where_clauses(wfcx, item.span, item.owner_id.def_id);
        Ok(())
    })
}

/// Checks where-clauses and inline bounds that are declared on `def_id`.
#[instrument(level = "debug", skip(wfcx))]
fn check_where_clauses<'tcx>(wfcx: &WfCheckingCtxt<'_, 'tcx>, span: Span, def_id: LocalDefId) {
    let infcx = wfcx.infcx;
    let tcx = wfcx.tcx();

    let predicates = tcx.predicates_of(def_id.to_def_id());
    let generics = tcx.generics_of(def_id);

    // Check that concrete defaults are well-formed. See test `type-check-defaults.rs`.
    // For example, this forbids the declaration:
    //
    //     struct Foo<T = Vec<[u32]>> { .. }
    //
    // Here, the default `Vec<[u32]>` is not WF because `[u32]: Sized` does not hold.
    for param in &generics.own_params {
        if let Some(default) = param.default_value(tcx).map(ty::EarlyBinder::instantiate_identity) {
            // Ignore dependent defaults -- that is, where the default of one type
            // parameter includes another (e.g., `<T, U = T>`). In those cases, we can't
            // be sure if it will error or not as user might always specify the other.
            // FIXME(generic_const_exprs): This is incorrect when dealing with unused const params.
            // E.g: `struct Foo<const N: usize, const M: usize = { 1 - 2 }>;`. Here, we should
            // eagerly error but we don't as we have `ConstKind::Unevaluated(.., [N, M])`.
            if !default.has_param() {
                wfcx.register_wf_obligation(
                    tcx.def_span(param.def_id),
                    matches!(param.kind, GenericParamDefKind::Type { .. })
                        .then(|| WellFormedLoc::Ty(param.def_id.expect_local())),
                    default.as_term().unwrap(),
                );
            } else {
                // If we've got a generic const parameter we still want to check its
                // type is correct in case both it and the param type are fully concrete.
                let GenericArgKind::Const(ct) = default.kind() else {
                    continue;
                };

                let ct_ty = match ct.kind() {
                    ty::ConstKind::Infer(_)
                    | ty::ConstKind::Placeholder(_)
                    | ty::ConstKind::Bound(_, _) => unreachable!(),
                    ty::ConstKind::Error(_) | ty::ConstKind::Expr(_) => continue,
                    ty::ConstKind::Value(cv) => cv.ty,
                    ty::ConstKind::Unevaluated(uv) => {
                        infcx.tcx.type_of(uv.def).instantiate(infcx.tcx, uv.args)
                    }
                    ty::ConstKind::Param(param_ct) => param_ct.find_ty_from_env(wfcx.param_env),
                };

                let param_ty = tcx.type_of(param.def_id).instantiate_identity();
                if !ct_ty.has_param() && !param_ty.has_param() {
                    let cause = traits::ObligationCause::new(
                        tcx.def_span(param.def_id),
                        wfcx.body_def_id,
                        ObligationCauseCode::WellFormed(None),
                    );
                    wfcx.register_obligation(Obligation::new(
                        tcx,
                        cause,
                        wfcx.param_env,
                        ty::ClauseKind::ConstArgHasType(ct, param_ty),
                    ));
                }
            }
        }
    }

    // Check that trait predicates are WF when params are instantiated with their defaults.
    // We don't want to overly constrain the predicates that may be written but we want to
    // catch cases where a default my never be applied such as `struct Foo<T: Copy = String>`.
    // Therefore we check if a predicate which contains a single type param
    // with a concrete default is WF with that default instantiated.
    // For more examples see tests `defaults-well-formedness.rs` and `type-check-defaults.rs`.
    //
    // First we build the defaulted generic parameters.
    let args = GenericArgs::for_item(tcx, def_id.to_def_id(), |param, _| {
        if param.index >= generics.parent_count as u32
            // If the param has a default, ...
            && let Some(default) = param.default_value(tcx).map(ty::EarlyBinder::instantiate_identity)
            // ... and it's not a dependent default, ...
            && !default.has_param()
        {
            // ... then instantiate it with the default.
            return default;
        }
        tcx.mk_param_from_def(param)
    });

    // Now we build the instantiated predicates.
    let default_obligations = predicates
        .predicates
        .iter()
        .flat_map(|&(pred, sp)| {
            #[derive(Default)]
            struct CountParams {
                params: FxHashSet<u32>,
            }
            impl<'tcx> ty::TypeVisitor<TyCtxt<'tcx>> for CountParams {
                type Result = ControlFlow<()>;
                fn visit_ty(&mut self, t: Ty<'tcx>) -> Self::Result {
                    if let ty::Param(param) = t.kind() {
                        self.params.insert(param.index);
                    }
                    t.super_visit_with(self)
                }

                fn visit_region(&mut self, _: ty::Region<'tcx>) -> Self::Result {
                    ControlFlow::Break(())
                }

                fn visit_const(&mut self, c: ty::Const<'tcx>) -> Self::Result {
                    if let ty::ConstKind::Param(param) = c.kind() {
                        self.params.insert(param.index);
                    }
                    c.super_visit_with(self)
                }
            }
            let mut param_count = CountParams::default();
            let has_region = pred.visit_with(&mut param_count).is_break();
            let instantiated_pred = ty::EarlyBinder::bind(pred).instantiate(tcx, args);
            // Don't check non-defaulted params, dependent defaults (including lifetimes)
            // or preds with multiple params.
            if instantiated_pred.has_non_region_param()
                || param_count.params.len() > 1
                || has_region
            {
                None
            } else if predicates.predicates.iter().any(|&(p, _)| p == instantiated_pred) {
                // Avoid duplication of predicates that contain no parameters, for example.
                None
            } else {
                Some((instantiated_pred, sp))
            }
        })
        .map(|(pred, sp)| {
            // Convert each of those into an obligation. So if you have
            // something like `struct Foo<T: Copy = String>`, we would
            // take that predicate `T: Copy`, instantiated with `String: Copy`
            // (actually that happens in the previous `flat_map` call),
            // and then try to prove it (in this case, we'll fail).
            //
            // Note the subtle difference from how we handle `predicates`
            // below: there, we are not trying to prove those predicates
            // to be *true* but merely *well-formed*.
            let pred = wfcx.normalize(sp, None, pred);
            let cause = traits::ObligationCause::new(
                sp,
                wfcx.body_def_id,
                ObligationCauseCode::WhereClause(def_id.to_def_id(), DUMMY_SP),
            );
            Obligation::new(tcx, cause, wfcx.param_env, pred)
        });

    let predicates = predicates.instantiate_identity(tcx);

    let predicates = wfcx.normalize(span, None, predicates);

    debug!(?predicates.predicates);
    assert_eq!(predicates.predicates.len(), predicates.spans.len());
    let wf_obligations = predicates.into_iter().flat_map(|(p, sp)| {
        traits::wf::clause_obligations(infcx, wfcx.param_env, wfcx.body_def_id, p, sp)
    });
    let obligations: Vec<_> = wf_obligations.chain(default_obligations).collect();
    wfcx.register_obligations(obligations);
}

#[instrument(level = "debug", skip(wfcx, span, hir_decl))]
fn check_fn_or_method<'tcx>(
    wfcx: &WfCheckingCtxt<'_, 'tcx>,
    span: Span,
    sig: ty::PolyFnSig<'tcx>,
    hir_decl: &hir::FnDecl<'_>,
    def_id: LocalDefId,
) {
    let tcx = wfcx.tcx();
    let mut sig = tcx.liberate_late_bound_regions(def_id.to_def_id(), sig);

    // Normalize the input and output types one at a time, using a different
    // `WellFormedLoc` for each. We cannot call `normalize_associated_types`
    // on the entire `FnSig`, since this would use the same `WellFormedLoc`
    // for each type, preventing the HIR wf check from generating
    // a nice error message.
    let arg_span =
        |idx| hir_decl.inputs.get(idx).map_or(hir_decl.output.span(), |arg: &hir::Ty<'_>| arg.span);

    sig.inputs_and_output =
        tcx.mk_type_list_from_iter(sig.inputs_and_output.iter().enumerate().map(|(idx, ty)| {
            wfcx.deeply_normalize(
                arg_span(idx),
                Some(WellFormedLoc::Param {
                    function: def_id,
                    // Note that the `param_idx` of the output type is
                    // one greater than the index of the last input type.
                    param_idx: idx,
                }),
                ty,
            )
        }));

    for (idx, ty) in sig.inputs_and_output.iter().enumerate() {
        wfcx.register_wf_obligation(
            arg_span(idx),
            Some(WellFormedLoc::Param { function: def_id, param_idx: idx }),
            ty.into(),
        );
    }

    check_where_clauses(wfcx, span, def_id);

    if sig.abi == ExternAbi::RustCall {
        let span = tcx.def_span(def_id);
        let has_implicit_self = hir_decl.implicit_self != hir::ImplicitSelfKind::None;
        let mut inputs = sig.inputs().iter().skip(if has_implicit_self { 1 } else { 0 });
        // Check that the argument is a tuple and is sized
        if let Some(ty) = inputs.next() {
            wfcx.register_bound(
                ObligationCause::new(span, wfcx.body_def_id, ObligationCauseCode::RustCall),
                wfcx.param_env,
                *ty,
                tcx.require_lang_item(hir::LangItem::Tuple, span),
            );
            wfcx.register_bound(
                ObligationCause::new(span, wfcx.body_def_id, ObligationCauseCode::RustCall),
                wfcx.param_env,
                *ty,
                tcx.require_lang_item(hir::LangItem::Sized, span),
            );
        } else {
            tcx.dcx().span_err(
                hir_decl.inputs.last().map_or(span, |input| input.span),
                "functions with the \"rust-call\" ABI must take a single non-self tuple argument",
            );
        }
        // No more inputs other than the `self` type and the tuple type
        if inputs.next().is_some() {
            tcx.dcx().span_err(
                hir_decl.inputs.last().map_or(span, |input| input.span),
                "functions with the \"rust-call\" ABI must take a single non-self tuple argument",
            );
        }
    }

    // If the function has a body, additionally require that the return type is sized.
    check_sized_if_body(
        wfcx,
        def_id,
        sig.output(),
        match hir_decl.output {
            hir::FnRetTy::Return(ty) => Some(ty.span),
            hir::FnRetTy::DefaultReturn(_) => None,
        },
        ObligationCauseCode::SizedReturnType,
    );
}

fn check_sized_if_body<'tcx>(
    wfcx: &WfCheckingCtxt<'_, 'tcx>,
    def_id: LocalDefId,
    ty: Ty<'tcx>,
    maybe_span: Option<Span>,
    code: ObligationCauseCode<'tcx>,
) {
    let tcx = wfcx.tcx();
    if let Some(body) = tcx.hir_maybe_body_owned_by(def_id) {
        let span = maybe_span.unwrap_or(body.value.span);

        wfcx.register_bound(
            ObligationCause::new(span, def_id, code),
            wfcx.param_env,
            ty,
            tcx.require_lang_item(LangItem::Sized, span),
        );
    }
}

/// The `arbitrary_self_types_pointers` feature implies `arbitrary_self_types`.
#[derive(Clone, Copy, PartialEq)]
enum ArbitrarySelfTypesLevel {
    Basic,        // just arbitrary_self_types
    WithPointers, // both arbitrary_self_types and arbitrary_self_types_pointers
}

#[instrument(level = "debug", skip(wfcx))]
fn check_method_receiver<'tcx>(
    wfcx: &WfCheckingCtxt<'_, 'tcx>,
    fn_sig: &hir::FnSig<'_>,
    method: ty::AssocItem,
    self_ty: Ty<'tcx>,
) -> Result<(), ErrorGuaranteed> {
    let tcx = wfcx.tcx();

    if !method.is_method() {
        return Ok(());
    }

    let span = fn_sig.decl.inputs[0].span;

    let sig = tcx.fn_sig(method.def_id).instantiate_identity();
    let sig = tcx.liberate_late_bound_regions(method.def_id, sig);
    let sig = wfcx.normalize(span, None, sig);

    debug!("check_method_receiver: sig={:?}", sig);

    let self_ty = wfcx.normalize(span, None, self_ty);

    let receiver_ty = sig.inputs()[0];
    let receiver_ty = wfcx.normalize(span, None, receiver_ty);

    // If the receiver already has errors reported, consider it valid to avoid
    // unnecessary errors (#58712).
    if receiver_ty.references_error() {
        return Ok(());
    }

    let arbitrary_self_types_level = if tcx.features().arbitrary_self_types_pointers() {
        Some(ArbitrarySelfTypesLevel::WithPointers)
    } else if tcx.features().arbitrary_self_types() {
        Some(ArbitrarySelfTypesLevel::Basic)
    } else {
        None
    };
    let generics = tcx.generics_of(method.def_id);

    let receiver_validity =
        receiver_is_valid(wfcx, span, receiver_ty, self_ty, arbitrary_self_types_level, generics);
    if let Err(receiver_validity_err) = receiver_validity {
        return Err(match arbitrary_self_types_level {
            // Wherever possible, emit a message advising folks that the features
            // `arbitrary_self_types` or `arbitrary_self_types_pointers` might
            // have helped.
            None if receiver_is_valid(
                wfcx,
                span,
                receiver_ty,
                self_ty,
                Some(ArbitrarySelfTypesLevel::Basic),
                generics,
            )
            .is_ok() =>
            {
                // Report error; would have worked with `arbitrary_self_types`.
                feature_err(
                    &tcx.sess,
                    sym::arbitrary_self_types,
                    span,
                    format!(
                        "`{receiver_ty}` cannot be used as the type of `self` without \
                            the `arbitrary_self_types` feature",
                    ),
                )
                .with_help(fluent::hir_analysis_invalid_receiver_ty_help)
                .emit()
            }
            None | Some(ArbitrarySelfTypesLevel::Basic)
                if receiver_is_valid(
                    wfcx,
                    span,
                    receiver_ty,
                    self_ty,
                    Some(ArbitrarySelfTypesLevel::WithPointers),
                    generics,
                )
                .is_ok() =>
            {
                // Report error; would have worked with `arbitrary_self_types_pointers`.
                feature_err(
                    &tcx.sess,
                    sym::arbitrary_self_types_pointers,
                    span,
                    format!(
                        "`{receiver_ty}` cannot be used as the type of `self` without \
                            the `arbitrary_self_types_pointers` feature",
                    ),
                )
                .with_help(fluent::hir_analysis_invalid_receiver_ty_help)
                .emit()
            }
            _ =>
            // Report error; would not have worked with `arbitrary_self_types[_pointers]`.
            {
                match receiver_validity_err {
                    ReceiverValidityError::DoesNotDeref if arbitrary_self_types_level.is_some() => {
                        let hint = match receiver_ty
                            .builtin_deref(false)
                            .unwrap_or(receiver_ty)
                            .ty_adt_def()
                            .and_then(|adt_def| tcx.get_diagnostic_name(adt_def.did()))
                        {
                            Some(sym::RcWeak | sym::ArcWeak) => Some(InvalidReceiverTyHint::Weak),
                            Some(sym::NonNull) => Some(InvalidReceiverTyHint::NonNull),
                            _ => None,
                        };

                        tcx.dcx().emit_err(errors::InvalidReceiverTy { span, receiver_ty, hint })
                    }
                    ReceiverValidityError::DoesNotDeref => {
                        tcx.dcx().emit_err(errors::InvalidReceiverTyNoArbitrarySelfTypes {
                            span,
                            receiver_ty,
                        })
                    }
                    ReceiverValidityError::MethodGenericParamUsed => {
                        tcx.dcx().emit_err(errors::InvalidGenericReceiverTy { span, receiver_ty })
                    }
                }
            }
        });
    }
    Ok(())
}

/// Error cases which may be returned from `receiver_is_valid`. These error
/// cases are generated in this function as they may be unearthed as we explore
/// the `autoderef` chain, but they're converted to diagnostics in the caller.
enum ReceiverValidityError {
    /// The self type does not get to the receiver type by following the
    /// autoderef chain.
    DoesNotDeref,
    /// A type was found which is a method type parameter, and that's not allowed.
    MethodGenericParamUsed,
}

/// Confirms that a type is not a type parameter referring to one of the
/// method's type params.
fn confirm_type_is_not_a_method_generic_param(
    ty: Ty<'_>,
    method_generics: &ty::Generics,
) -> Result<(), ReceiverValidityError> {
    if let ty::Param(param) = ty.kind() {
        if (param.index as usize) >= method_generics.parent_count {
            return Err(ReceiverValidityError::MethodGenericParamUsed);
        }
    }
    Ok(())
}

/// Returns whether `receiver_ty` would be considered a valid receiver type for `self_ty`. If
/// `arbitrary_self_types` is enabled, `receiver_ty` must transitively deref to `self_ty`, possibly
/// through a `*const/mut T` raw pointer if  `arbitrary_self_types_pointers` is also enabled.
/// If neither feature is enabled, the requirements are more strict: `receiver_ty` must implement
/// `Receiver` and directly implement `Deref<Target = self_ty>`.
///
/// N.B., there are cases this function returns `true` but causes an error to be emitted,
/// particularly when `receiver_ty` derefs to a type that is the same as `self_ty` but has the
/// wrong lifetime. Be careful of this if you are calling this function speculatively.
fn receiver_is_valid<'tcx>(
    wfcx: &WfCheckingCtxt<'_, 'tcx>,
    span: Span,
    receiver_ty: Ty<'tcx>,
    self_ty: Ty<'tcx>,
    arbitrary_self_types_enabled: Option<ArbitrarySelfTypesLevel>,
    method_generics: &ty::Generics,
) -> Result<(), ReceiverValidityError> {
    let infcx = wfcx.infcx;
    let tcx = wfcx.tcx();
    let cause =
        ObligationCause::new(span, wfcx.body_def_id, traits::ObligationCauseCode::MethodReceiver);

    // Special case `receiver == self_ty`, which doesn't necessarily require the `Receiver` lang item.
    if let Ok(()) = wfcx.infcx.commit_if_ok(|_| {
        let ocx = ObligationCtxt::new(wfcx.infcx);
        ocx.eq(&cause, wfcx.param_env, self_ty, receiver_ty)?;
        if ocx.select_all_or_error().is_empty() { Ok(()) } else { Err(NoSolution) }
    }) {
        return Ok(());
    }

    confirm_type_is_not_a_method_generic_param(receiver_ty, method_generics)?;

    let mut autoderef = Autoderef::new(infcx, wfcx.param_env, wfcx.body_def_id, span, receiver_ty);

    // The `arbitrary_self_types` feature allows custom smart pointer
    // types to be method receivers, as identified by following the Receiver<Target=T>
    // chain.
    if arbitrary_self_types_enabled.is_some() {
        autoderef = autoderef.use_receiver_trait();
    }

    // The `arbitrary_self_types_pointers` feature allows raw pointer receivers like `self: *const Self`.
    if arbitrary_self_types_enabled == Some(ArbitrarySelfTypesLevel::WithPointers) {
        autoderef = autoderef.include_raw_pointers();
    }

    // Keep dereferencing `receiver_ty` until we get to `self_ty`.
    while let Some((potential_self_ty, _)) = autoderef.next() {
        debug!(
            "receiver_is_valid: potential self type `{:?}` to match `{:?}`",
            potential_self_ty, self_ty
        );

        confirm_type_is_not_a_method_generic_param(potential_self_ty, method_generics)?;

        // Check if the self type unifies. If it does, then commit the result
        // since it may have region side-effects.
        if let Ok(()) = wfcx.infcx.commit_if_ok(|_| {
            let ocx = ObligationCtxt::new(wfcx.infcx);
            ocx.eq(&cause, wfcx.param_env, self_ty, potential_self_ty)?;
            if ocx.select_all_or_error().is_empty() { Ok(()) } else { Err(NoSolution) }
        }) {
            wfcx.register_obligations(autoderef.into_obligations());
            return Ok(());
        }

        // Without `feature(arbitrary_self_types)`, we require that each step in the
        // deref chain implement `LegacyReceiver`.
        if arbitrary_self_types_enabled.is_none() {
            let legacy_receiver_trait_def_id =
                tcx.require_lang_item(LangItem::LegacyReceiver, span);
            if !legacy_receiver_is_implemented(
                wfcx,
                legacy_receiver_trait_def_id,
                cause.clone(),
                potential_self_ty,
            ) {
                // We cannot proceed.
                break;
            }

            // Register the bound, in case it has any region side-effects.
            wfcx.register_bound(
                cause.clone(),
                wfcx.param_env,
                potential_self_ty,
                legacy_receiver_trait_def_id,
            );
        }
    }

    debug!("receiver_is_valid: type `{:?}` does not deref to `{:?}`", receiver_ty, self_ty);
    Err(ReceiverValidityError::DoesNotDeref)
}

fn legacy_receiver_is_implemented<'tcx>(
    wfcx: &WfCheckingCtxt<'_, 'tcx>,
    legacy_receiver_trait_def_id: DefId,
    cause: ObligationCause<'tcx>,
    receiver_ty: Ty<'tcx>,
) -> bool {
    let tcx = wfcx.tcx();
    let trait_ref = ty::TraitRef::new(tcx, legacy_receiver_trait_def_id, [receiver_ty]);

    let obligation = Obligation::new(tcx, cause, wfcx.param_env, trait_ref);

    if wfcx.infcx.predicate_must_hold_modulo_regions(&obligation) {
        true
    } else {
        debug!(
            "receiver_is_implemented: type `{:?}` does not implement `LegacyReceiver` trait",
            receiver_ty
        );
        false
    }
}

fn check_variances_for_type_defn<'tcx>(
    tcx: TyCtxt<'tcx>,
    item: &'tcx hir::Item<'tcx>,
    hir_generics: &hir::Generics<'tcx>,
) {
    match item.kind {
        ItemKind::Enum(..) | ItemKind::Struct(..) | ItemKind::Union(..) => {
            // Ok
        }
        ItemKind::TyAlias(..) => {
            assert!(
                tcx.type_alias_is_lazy(item.owner_id),
                "should not be computing variance of non-free type alias"
            );
        }
        kind => span_bug!(item.span, "cannot compute the variances of {kind:?}"),
    }

    let ty_predicates = tcx.predicates_of(item.owner_id);
    assert_eq!(ty_predicates.parent, None);
    let variances = tcx.variances_of(item.owner_id);

    let mut constrained_parameters: FxHashSet<_> = variances
        .iter()
        .enumerate()
        .filter(|&(_, &variance)| variance != ty::Bivariant)
        .map(|(index, _)| Parameter(index as u32))
        .collect();

    identify_constrained_generic_params(tcx, ty_predicates, None, &mut constrained_parameters);

    // Lazily calculated because it is only needed in case of an error.
    let explicitly_bounded_params = LazyCell::new(|| {
        let icx = crate::collect::ItemCtxt::new(tcx, item.owner_id.def_id);
        hir_generics
            .predicates
            .iter()
            .filter_map(|predicate| match predicate.kind {
                hir::WherePredicateKind::BoundPredicate(predicate) => {
                    match icx.lower_ty(predicate.bounded_ty).kind() {
                        ty::Param(data) => Some(Parameter(data.index)),
                        _ => None,
                    }
                }
                _ => None,
            })
            .collect::<FxHashSet<_>>()
    });

    let ty_generics = tcx.generics_of(item.owner_id);

    for (index, _) in variances.iter().enumerate() {
        let parameter = Parameter(index as u32);

        if constrained_parameters.contains(&parameter) {
            continue;
        }

        let ty_param = &ty_generics.own_params[index];
        let hir_param = &hir_generics.params[index];

        if ty_param.def_id != hir_param.def_id.into() {
            // Valid programs always have lifetimes before types in the generic parameter list.
            // ty_generics are normalized to be in this required order, and variances are built
            // from ty generics, not from hir generics. but we need hir generics to get
            // a span out.
            //
            // If they aren't in the same order, then the user has written invalid code, and already
            // got an error about it (or I'm wrong about this).
            tcx.dcx().span_delayed_bug(
                hir_param.span,
                "hir generics and ty generics in different order",
            );
            continue;
        }

        // Look for `ErrorGuaranteed` deeply within this type.
        if let ControlFlow::Break(ErrorGuaranteed { .. }) = tcx
            .type_of(item.owner_id)
            .instantiate_identity()
            .visit_with(&mut HasErrorDeep { tcx, seen: Default::default() })
        {
            continue;
        }

        match hir_param.name {
            hir::ParamName::Error(_) => {
                // Don't report a bivariance error for a lifetime that isn't
                // even valid to name.
            }
            _ => {
                let has_explicit_bounds = explicitly_bounded_params.contains(&parameter);
                report_bivariance(tcx, hir_param, has_explicit_bounds, item);
            }
        }
    }
}

/// Look for `ErrorGuaranteed` deeply within structs' (unsubstituted) fields.
struct HasErrorDeep<'tcx> {
    tcx: TyCtxt<'tcx>,
    seen: FxHashSet<DefId>,
}
impl<'tcx> TypeVisitor<TyCtxt<'tcx>> for HasErrorDeep<'tcx> {
    type Result = ControlFlow<ErrorGuaranteed>;

    fn visit_ty(&mut self, ty: Ty<'tcx>) -> Self::Result {
        match *ty.kind() {
            ty::Adt(def, _) => {
                if self.seen.insert(def.did()) {
                    for field in def.all_fields() {
                        self.tcx.type_of(field.did).instantiate_identity().visit_with(self)?;
                    }
                }
            }
            ty::Error(guar) => return ControlFlow::Break(guar),
            _ => {}
        }
        ty.super_visit_with(self)
    }

    fn visit_region(&mut self, r: ty::Region<'tcx>) -> Self::Result {
        if let Err(guar) = r.error_reported() {
            ControlFlow::Break(guar)
        } else {
            ControlFlow::Continue(())
        }
    }

    fn visit_const(&mut self, c: ty::Const<'tcx>) -> Self::Result {
        if let Err(guar) = c.error_reported() {
            ControlFlow::Break(guar)
        } else {
            ControlFlow::Continue(())
        }
    }
}

fn report_bivariance<'tcx>(
    tcx: TyCtxt<'tcx>,
    param: &'tcx hir::GenericParam<'tcx>,
    has_explicit_bounds: bool,
    item: &'tcx hir::Item<'tcx>,
) -> ErrorGuaranteed {
    let param_name = param.name.ident();

    let help = match item.kind {
        ItemKind::Enum(..) | ItemKind::Struct(..) | ItemKind::Union(..) => {
            if let Some(def_id) = tcx.lang_items().phantom_data() {
                errors::UnusedGenericParameterHelp::Adt {
                    param_name,
                    phantom_data: tcx.def_path_str(def_id),
                }
            } else {
                errors::UnusedGenericParameterHelp::AdtNoPhantomData { param_name }
            }
        }
        ItemKind::TyAlias(..) => errors::UnusedGenericParameterHelp::TyAlias { param_name },
        item_kind => bug!("report_bivariance: unexpected item kind: {item_kind:?}"),
    };

    let mut usage_spans = vec![];
    intravisit::walk_item(
        &mut CollectUsageSpans { spans: &mut usage_spans, param_def_id: param.def_id.to_def_id() },
        item,
    );

    if !usage_spans.is_empty() {
        // First, check if the ADT/LTA is (probably) cyclical. We say probably here, since we're
        // not actually looking into substitutions, just walking through fields / the "RHS".
        // We don't recurse into the hidden types of opaques or anything else fancy.
        let item_def_id = item.owner_id.to_def_id();
        let is_probably_cyclical =
            IsProbablyCyclical { tcx, item_def_id, seen: Default::default() }
                .visit_def(item_def_id)
                .is_break();
        // If the ADT/LTA is cyclical, then if at least one usage of the type parameter or
        // the `Self` alias is present in the, then it's probably a cyclical struct/ type
        // alias, and we should call those parameter usages recursive rather than just saying
        // they're unused...
        //
        // We currently report *all* of the parameter usages, since computing the exact
        // subset is very involved, and the fact we're mentioning recursion at all is
        // likely to guide the user in the right direction.
        if is_probably_cyclical {
            return tcx.dcx().emit_err(errors::RecursiveGenericParameter {
                spans: usage_spans,
                param_span: param.span,
                param_name,
                param_def_kind: tcx.def_descr(param.def_id.to_def_id()),
                help,
                note: (),
            });
        }
    }

    let const_param_help =
        matches!(param.kind, hir::GenericParamKind::Type { .. } if !has_explicit_bounds);

    let mut diag = tcx.dcx().create_err(errors::UnusedGenericParameter {
        span: param.span,
        param_name,
        param_def_kind: tcx.def_descr(param.def_id.to_def_id()),
        usage_spans,
        help,
        const_param_help,
    });
    diag.code(E0392);
    diag.emit()
}

/// Detects cases where an ADT/LTA is trivially cyclical -- we want to detect this so
/// we only mention that its parameters are used cyclically if the ADT/LTA is truly
/// cyclical.
///
/// Notably, we don't consider substitutions here, so this may have false positives.
struct IsProbablyCyclical<'tcx> {
    tcx: TyCtxt<'tcx>,
    item_def_id: DefId,
    seen: FxHashSet<DefId>,
}

impl<'tcx> IsProbablyCyclical<'tcx> {
    fn visit_def(&mut self, def_id: DefId) -> ControlFlow<(), ()> {
        match self.tcx.def_kind(def_id) {
            DefKind::Struct | DefKind::Enum | DefKind::Union => {
                self.tcx.adt_def(def_id).all_fields().try_for_each(|field| {
                    self.tcx.type_of(field.did).instantiate_identity().visit_with(self)
                })
            }
            DefKind::TyAlias if self.tcx.type_alias_is_lazy(def_id) => {
                self.tcx.type_of(def_id).instantiate_identity().visit_with(self)
            }
            _ => ControlFlow::Continue(()),
        }
    }
}

impl<'tcx> TypeVisitor<TyCtxt<'tcx>> for IsProbablyCyclical<'tcx> {
    type Result = ControlFlow<(), ()>;

    fn visit_ty(&mut self, ty: Ty<'tcx>) -> ControlFlow<(), ()> {
        let def_id = match ty.kind() {
            ty::Adt(adt_def, _) => Some(adt_def.did()),
            ty::Alias(ty::Free, alias_ty) => Some(alias_ty.def_id),
            _ => None,
        };
        if let Some(def_id) = def_id {
            if def_id == self.item_def_id {
                return ControlFlow::Break(());
            }
            if self.seen.insert(def_id) {
                self.visit_def(def_id)?;
            }
        }
        ty.super_visit_with(self)
    }
}

/// Collect usages of the `param_def_id` and `Res::SelfTyAlias` in the HIR.
///
/// This is used to report places where the user has used parameters in a
/// non-variance-constraining way for better bivariance errors.
struct CollectUsageSpans<'a> {
    spans: &'a mut Vec<Span>,
    param_def_id: DefId,
}

impl<'tcx> Visitor<'tcx> for CollectUsageSpans<'_> {
    type Result = ();

    fn visit_generics(&mut self, _g: &'tcx rustc_hir::Generics<'tcx>) -> Self::Result {
        // Skip the generics. We only care about fields, not where clause/param bounds.
    }

    fn visit_ty(&mut self, t: &'tcx hir::Ty<'tcx, AmbigArg>) -> Self::Result {
        if let hir::TyKind::Path(hir::QPath::Resolved(None, qpath)) = t.kind {
            if let Res::Def(DefKind::TyParam, def_id) = qpath.res
                && def_id == self.param_def_id
            {
                self.spans.push(t.span);
                return;
            } else if let Res::SelfTyAlias { .. } = qpath.res {
                self.spans.push(t.span);
                return;
            }
        }
        intravisit::walk_ty(self, t);
    }
}

impl<'tcx> WfCheckingCtxt<'_, 'tcx> {
    /// Feature gates RFC 2056 -- trivial bounds, checking for global bounds that
    /// aren't true.
    #[instrument(level = "debug", skip(self))]
    fn check_false_global_bounds(&mut self) {
        let tcx = self.ocx.infcx.tcx;
        let mut span = self.span;
        let empty_env = ty::ParamEnv::empty();

        let predicates_with_span = tcx.predicates_of(self.body_def_id).predicates.iter().copied();
        // Check elaborated bounds.
        let implied_obligations = traits::elaborate(tcx, predicates_with_span);

        for (pred, obligation_span) in implied_obligations {
            // We lower empty bounds like `Vec<dyn Copy>:` as
            // `WellFormed(Vec<dyn Copy>)`, which will later get checked by
            // regular WF checking
            if let ty::ClauseKind::WellFormed(..) = pred.kind().skip_binder() {
                continue;
            }
            // Match the existing behavior.
            if pred.is_global() && !pred.has_type_flags(TypeFlags::HAS_BINDER_VARS) {
                let pred = self.normalize(span, None, pred);

                // only use the span of the predicate clause (#90869)
                let hir_node = tcx.hir_node_by_def_id(self.body_def_id);
                if let Some(hir::Generics { predicates, .. }) = hir_node.generics() {
                    span = predicates
                        .iter()
                        // There seems to be no better way to find out which predicate we are in
                        .find(|pred| pred.span.contains(obligation_span))
                        .map(|pred| pred.span)
                        .unwrap_or(obligation_span);
                }

                let obligation = Obligation::new(
                    tcx,
                    traits::ObligationCause::new(
                        span,
                        self.body_def_id,
                        ObligationCauseCode::TrivialBound,
                    ),
                    empty_env,
                    pred,
                );
                self.ocx.register_obligation(obligation);
            }
        }
    }
}

fn check_type_wf(tcx: TyCtxt<'_>, (): ()) -> Result<(), ErrorGuaranteed> {
    let items = tcx.hir_crate_items(());
    let res = items
        .par_items(|item| tcx.ensure_ok().check_well_formed(item.owner_id.def_id))
        .and(items.par_impl_items(|item| tcx.ensure_ok().check_well_formed(item.owner_id.def_id)))
        .and(items.par_trait_items(|item| tcx.ensure_ok().check_well_formed(item.owner_id.def_id)))
        .and(
            items.par_foreign_items(|item| tcx.ensure_ok().check_well_formed(item.owner_id.def_id)),
        )
        .and(items.par_nested_bodies(|item| tcx.ensure_ok().check_well_formed(item)))
        .and(items.par_opaques(|item| tcx.ensure_ok().check_well_formed(item)));
    super::entry::check_for_entry_fn(tcx);

    res
}

fn lint_redundant_lifetimes<'tcx>(
    tcx: TyCtxt<'tcx>,
    owner_id: LocalDefId,
    outlives_env: &OutlivesEnvironment<'tcx>,
) {
    let def_kind = tcx.def_kind(owner_id);
    match def_kind {
        DefKind::Struct
        | DefKind::Union
        | DefKind::Enum
        | DefKind::Trait
        | DefKind::TraitAlias
        | DefKind::Fn
        | DefKind::Const
        | DefKind::Impl { of_trait: _ } => {
            // Proceed
        }
        DefKind::AssocFn | DefKind::AssocTy | DefKind::AssocConst => {
            let parent_def_id = tcx.local_parent(owner_id);
            if matches!(tcx.def_kind(parent_def_id), DefKind::Impl { of_trait: true }) {
                // Don't check for redundant lifetimes for associated items of trait
                // implementations, since the signature is required to be compatible
                // with the trait, even if the implementation implies some lifetimes
                // are redundant.
                return;
            }
        }
        DefKind::Mod
        | DefKind::Variant
        | DefKind::TyAlias
        | DefKind::ForeignTy
        | DefKind::TyParam
        | DefKind::ConstParam
        | DefKind::Static { .. }
        | DefKind::Ctor(_, _)
        | DefKind::Macro(_)
        | DefKind::ExternCrate
        | DefKind::Use
        | DefKind::ForeignMod
        | DefKind::AnonConst
        | DefKind::InlineConst
        | DefKind::OpaqueTy
        | DefKind::Field
        | DefKind::LifetimeParam
        | DefKind::GlobalAsm
        | DefKind::Closure
        | DefKind::SyntheticCoroutineBody => return,
    }

    // The ordering of this lifetime map is a bit subtle.
    //
    // Specifically, we want to find a "candidate" lifetime that precedes a "victim" lifetime,
    // where we can prove that `'candidate = 'victim`.
    //
    // `'static` must come first in this list because we can never replace `'static` with
    // something else, but if we find some lifetime `'a` where `'a = 'static`, we want to
    // suggest replacing `'a` with `'static`.
    let mut lifetimes = vec![tcx.lifetimes.re_static];
    lifetimes.extend(
        ty::GenericArgs::identity_for_item(tcx, owner_id).iter().filter_map(|arg| arg.as_region()),
    );
    // If we are in a function, add its late-bound lifetimes too.
    if matches!(def_kind, DefKind::Fn | DefKind::AssocFn) {
        for (idx, var) in
            tcx.fn_sig(owner_id).instantiate_identity().bound_vars().iter().enumerate()
        {
            let ty::BoundVariableKind::Region(kind) = var else { continue };
            let kind = ty::LateParamRegionKind::from_bound(ty::BoundVar::from_usize(idx), kind);
            lifetimes.push(ty::Region::new_late_param(tcx, owner_id.to_def_id(), kind));
        }
    }
    lifetimes.retain(|candidate| candidate.has_name());

    // Keep track of lifetimes which have already been replaced with other lifetimes.
    // This makes sure that if `'a = 'b = 'c`, we don't say `'c` should be replaced by
    // both `'a` and `'b`.
    let mut shadowed = FxHashSet::default();

    for (idx, &candidate) in lifetimes.iter().enumerate() {
        // Don't suggest removing a lifetime twice. We only need to check this
        // here and not up in the `victim` loop because equality is transitive,
        // so if A = C and B = C, then A must = B, so it'll be shadowed too in
        // A's victim loop.
        if shadowed.contains(&candidate) {
            continue;
        }

        for &victim in &lifetimes[(idx + 1)..] {
            // All region parameters should have a `DefId` available as:
            // - Late-bound parameters should be of the`BrNamed` variety,
            // since we get these signatures straight from `hir_lowering`.
            // - Early-bound parameters unconditionally have a `DefId` available.
            //
            // Any other regions (ReError/ReStatic/etc.) shouldn't matter, since we
            // can't really suggest to remove them.
            let Some(def_id) = victim.opt_param_def_id(tcx, owner_id.to_def_id()) else {
                continue;
            };

            // Do not rename lifetimes not local to this item since they'll overlap
            // with the lint running on the parent. We still want to consider parent
            // lifetimes which make child lifetimes redundant, otherwise we would
            // have truncated the `identity_for_item` args above.
            if tcx.parent(def_id) != owner_id.to_def_id() {
                continue;
            }

            // If `candidate <: victim` and `victim <: candidate`, then they're equal.
            if outlives_env.free_region_map().sub_free_regions(tcx, candidate, victim)
                && outlives_env.free_region_map().sub_free_regions(tcx, victim, candidate)
            {
                shadowed.insert(victim);
                tcx.emit_node_span_lint(
                    rustc_lint_defs::builtin::REDUNDANT_LIFETIMES,
                    tcx.local_def_id_to_hir_id(def_id.expect_local()),
                    tcx.def_span(def_id),
                    RedundantLifetimeArgsLint { candidate, victim },
                );
            }
        }
    }
}

#[derive(LintDiagnostic)]
#[diag(hir_analysis_redundant_lifetime_args)]
#[note]
struct RedundantLifetimeArgsLint<'tcx> {
    /// The lifetime we have found to be redundant.
    victim: ty::Region<'tcx>,
    // The lifetime we can replace the victim with.
    candidate: ty::Region<'tcx>,
}

pub fn provide(providers: &mut Providers) {
    *providers = Providers { check_type_wf, check_well_formed, ..*providers };
}
