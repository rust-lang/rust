use crate::autoderef::Autoderef;
use crate::constrained_generic_params::{identify_constrained_generic_params, Parameter};
use crate::errors;

use rustc_ast as ast;
use rustc_data_structures::fx::{FxHashMap, FxHashSet, FxIndexSet};
use rustc_errors::{codes::*, pluralize, struct_span_code_err, Applicability, ErrorGuaranteed};
use rustc_hir as hir;
use rustc_hir::def_id::{DefId, LocalDefId, LocalModDefId};
use rustc_hir::lang_items::LangItem;
use rustc_hir::ItemKind;
use rustc_infer::infer::outlives::env::OutlivesEnvironment;
use rustc_infer::infer::{self, InferCtxt, TyCtxtInferExt};
use rustc_middle::query::Providers;
use rustc_middle::ty::print::with_no_trimmed_paths;
use rustc_middle::ty::trait_def::TraitSpecializationKind;
use rustc_middle::ty::{
    self, AdtKind, GenericParamDefKind, ToPredicate, Ty, TyCtxt, TypeFoldable, TypeSuperVisitable,
    TypeVisitable, TypeVisitableExt, TypeVisitor,
};
use rustc_middle::ty::{GenericArgKind, GenericArgs};
use rustc_session::parse::feature_err;
use rustc_span::symbol::{sym, Ident};
use rustc_span::{Span, DUMMY_SP};
use rustc_target::spec::abi::Abi;
use rustc_trait_selection::regions::InferCtxtRegionExt;
use rustc_trait_selection::traits::error_reporting::TypeErrCtxtExt;
use rustc_trait_selection::traits::misc::{
    type_allowed_to_implement_const_param_ty, ConstParamTyImplementationError,
};
use rustc_trait_selection::traits::outlives_bounds::InferCtxtExt as _;
use rustc_trait_selection::traits::query::evaluate_obligation::InferCtxtExt as _;
use rustc_trait_selection::traits::{
    self, ObligationCause, ObligationCauseCode, ObligationCtxt, WellFormedLoc,
};
use rustc_type_ir::TypeFlags;

use std::cell::LazyCell;
use std::ops::{ControlFlow, Deref};

pub(super) struct WfCheckingCtxt<'a, 'tcx> {
    pub(super) ocx: ObligationCtxt<'a, 'tcx>,
    span: Span,
    body_def_id: LocalDefId,
    param_env: ty::ParamEnv<'tcx>,
}
impl<'a, 'tcx> Deref for WfCheckingCtxt<'a, 'tcx> {
    type Target = ObligationCtxt<'a, 'tcx>;
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

    fn register_wf_obligation(
        &self,
        span: Span,
        loc: Option<WellFormedLoc>,
        arg: ty::GenericArg<'tcx>,
    ) {
        let cause = traits::ObligationCause::new(
            span,
            self.body_def_id,
            ObligationCauseCode::WellFormed(loc),
        );
        self.ocx.register_obligation(traits::Obligation::new(
            self.tcx(),
            cause,
            self.param_env,
            ty::Binder::dummy(ty::PredicateKind::Clause(ty::ClauseKind::WellFormed(arg))),
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
    let infcx = &tcx.infer_ctxt().build();
    let ocx = ObligationCtxt::new(infcx);

    let mut wfcx = WfCheckingCtxt { ocx, span, body_def_id, param_env };

    if !tcx.features().trivial_bounds {
        wfcx.check_false_global_bounds()
    }
    f(&mut wfcx)?;

    let assumed_wf_types = wfcx.ocx.assumed_wf_types_and_report_errors(param_env, body_def_id)?;

    let errors = wfcx.select_all_or_error();
    if !errors.is_empty() {
        let err = infcx.err_ctxt().report_fulfillment_errors(errors);
        if tcx.dcx().has_errors().is_some() {
            return Err(err);
        } else {
            // HACK(oli-obk): tests/ui/specialization/min_specialization/specialize_on_type_error.rs
            // causes an delayed bug during normalization, without reporting an error, so we need
            // to act as if no error happened, in order to let our callers continue and report an
            // error later in check_impl_items_against_trait.
            return Ok(());
        }
    }

    debug!(?assumed_wf_types);

    let infcx_compat = infcx.fork();

    // We specifically want to call the non-compat version of `implied_bounds_tys`; we do this always.
    let implied_bounds =
        infcx.implied_bounds_tys_compat(param_env, body_def_id, &assumed_wf_types, false);
    let outlives_env = OutlivesEnvironment::with_bounds(param_env, implied_bounds);

    let errors = infcx.resolve_regions(&outlives_env);
    if errors.is_empty() {
        return Ok(());
    }

    let is_bevy = 'is_bevy: {
        // We don't want to emit this for dependents of Bevy, for now.
        // See #119956
        let is_bevy_paramset = |def: ty::AdtDef<'_>| {
            let adt_did = with_no_trimmed_paths!(infcx.tcx.def_path_str(def.0.did));
            adt_did.contains("ParamSet")
        };
        for ty in assumed_wf_types.iter() {
            match ty.kind() {
                ty::Adt(def, _) => {
                    if is_bevy_paramset(*def) {
                        break 'is_bevy true;
                    }
                }
                ty::Ref(_, ty, _) => match ty.kind() {
                    ty::Adt(def, _) => {
                        if is_bevy_paramset(*def) {
                            break 'is_bevy true;
                        }
                    }
                    _ => {}
                },
                _ => {}
            }
        }
        false
    };

    // If we have set `no_implied_bounds_compat`, then do not attempt compatibility.
    // We could also just always enter if `is_bevy`, and call `implied_bounds_tys`,
    // but that does result in slightly more work when this option is set and
    // just obscures what we mean here anyways. Let's just be explicit.
    if is_bevy && !infcx.tcx.sess.opts.unstable_opts.no_implied_bounds_compat {
        let implied_bounds =
            infcx_compat.implied_bounds_tys_compat(param_env, body_def_id, &assumed_wf_types, true);
        let outlives_env = OutlivesEnvironment::with_bounds(param_env, implied_bounds);
        let errors_compat = infcx_compat.resolve_regions(&outlives_env);
        if errors_compat.is_empty() {
            Ok(())
        } else {
            Err(infcx_compat.err_ctxt().report_region_errors(body_def_id, &errors_compat))
        }
    } else {
        Err(infcx.err_ctxt().report_region_errors(body_def_id, &errors))
    }
}

fn check_well_formed(tcx: TyCtxt<'_>, def_id: hir::OwnerId) -> Result<(), ErrorGuaranteed> {
    let node = tcx.hir_owner_node(def_id);
    let mut res = match node {
        hir::OwnerNode::Crate(_) => bug!("check_well_formed cannot be applied to the crate root"),
        hir::OwnerNode::Item(item) => check_item(tcx, item),
        hir::OwnerNode::TraitItem(item) => check_trait_item(tcx, item),
        hir::OwnerNode::ImplItem(item) => check_impl_item(tcx, item),
        hir::OwnerNode::ForeignItem(item) => check_foreign_item(tcx, item),
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
/// struct Ref<'a, T> { x: &'a T }
/// ```
///
/// because the type did not declare that `T:'a`.
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
                .is_some_and(|header| tcx.trait_is_auto(header.skip_binder().trait_ref.def_id));
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
            match header.map(|h| h.skip_binder().polarity) {
                // `None` means this is an inherent impl
                Some(ty::ImplPolarity::Positive) | None => {
                    res = res.and(check_impl(tcx, item, impl_.self_ty, &impl_.of_trait));
                }
                Some(ty::ImplPolarity::Negative) => {
                    let ast::ImplPolarity::Negative(span) = impl_.polarity else {
                        bug!("impl_polarity query disagrees with impl's polarity in AST");
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
        hir::ItemKind::Fn(ref sig, ..) => {
            check_item_fn(tcx, def_id, item.ident, item.span, sig.decl)
        }
        hir::ItemKind::Static(ty, ..) => {
            check_item_type(tcx, def_id, ty.span, UnsizedHandling::Forbid)
        }
        hir::ItemKind::Const(ty, ..) => {
            check_item_type(tcx, def_id, ty.span, UnsizedHandling::Forbid)
        }
        hir::ItemKind::Struct(_, ast_generics) => {
            let res = check_type_defn(tcx, item, false);
            check_variances_for_type_defn(tcx, item, ast_generics);
            res
        }
        hir::ItemKind::Union(_, ast_generics) => {
            let res = check_type_defn(tcx, item, true);
            check_variances_for_type_defn(tcx, item, ast_generics);
            res
        }
        hir::ItemKind::Enum(_, ast_generics) => {
            let res = check_type_defn(tcx, item, true);
            check_variances_for_type_defn(tcx, item, ast_generics);
            res
        }
        hir::ItemKind::Trait(..) => check_trait(tcx, item),
        hir::ItemKind::TraitAlias(..) => check_trait(tcx, item),
        // `ForeignItem`s are handled separately.
        hir::ItemKind::ForeignMod { .. } => Ok(()),
        hir::ItemKind::TyAlias(hir_ty, ast_generics) => {
            if tcx.type_alias_is_lazy(item.owner_id) {
                // Bounds of lazy type aliases and of eager ones that contain opaque types are respected.
                // E.g: `type X = impl Trait;`, `type X = (impl Trait, Y);`.
                let res = check_item_type(tcx, def_id, hir_ty.span, UnsizedHandling::Allow);
                check_variances_for_type_defn(tcx, item, ast_generics);
                res
            } else {
                Ok(())
            }
        }
        _ => Ok(()),
    };

    crate::check::check::check_item_type(tcx, def_id);

    res
}

fn check_foreign_item(tcx: TyCtxt<'_>, item: &hir::ForeignItem<'_>) -> Result<(), ErrorGuaranteed> {
    let def_id = item.owner_id.def_id;

    debug!(
        ?item.owner_id,
        item.name = ? tcx.def_path_str(def_id)
    );

    match item.kind {
        hir::ForeignItemKind::Fn(decl, ..) => {
            check_item_fn(tcx, def_id, item.ident, item.span, decl)
        }
        hir::ForeignItemKind::Static(ty, ..) => {
            check_item_type(tcx, def_id, ty.span, UnsizedHandling::AllowIfForeignTail)
        }
        hir::ForeignItemKind::Type => Ok(()),
    }
}

fn check_trait_item(
    tcx: TyCtxt<'_>,
    trait_item: &hir::TraitItem<'_>,
) -> Result<(), ErrorGuaranteed> {
    let def_id = trait_item.owner_id.def_id;

    let (method_sig, span) = match trait_item.kind {
        hir::TraitItemKind::Fn(ref sig, _) => (Some(sig), trait_item.span),
        hir::TraitItemKind::Type(_bounds, Some(ty)) => (None, ty.span),
        _ => (None, trait_item.span),
    };
    check_object_unsafe_self_trait_by_name(tcx, trait_item);
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
    let mut required_bounds_by_item = FxHashMap::default();
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
            if gat_item.kind != ty::AssocKind::Type {
                continue;
            }
            let gat_generics = tcx.generics_of(gat_def_id);
            // FIXME(jackh726): we can also warn in the more general case
            if gat_generics.params.is_empty() {
                continue;
            }

            // Gather the bounds with which all other items inside of this trait constrain the GAT.
            // This is calculated by taking the intersection of the bounds that each item
            // constrains the GAT with individually.
            let mut new_required_bounds: Option<FxHashSet<ty::Clause<'_>>> = None;
            for item in associated_items.in_definition_order() {
                let item_def_id = item.def_id.expect_local();
                // Skip our own GAT, since it does not constrain itself at all.
                if item_def_id == gat_def_id {
                    continue;
                }

                let param_env = tcx.param_env(item_def_id);

                let item_required_bounds = match tcx.associated_item(item_def_id).kind {
                    // In our example, this corresponds to `into_iter` method
                    ty::AssocKind::Fn => {
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
                    ty::AssocKind::Type => {
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
                                .instantiate_identity_iter_copied()
                                .collect::<Vec<_>>(),
                            &FxIndexSet::default(),
                            gat_def_id,
                            gat_generics,
                        )
                    }
                    ty::AssocKind::Const => None,
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

        let gat_item_hir = tcx.hir().expect_trait_item(gat_def_id);
        debug!(?required_bounds);
        let param_env = tcx.param_env(gat_def_id);

        let mut unsatisfied_bounds: Vec<_> = required_bounds
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

        // We sort so that order is predictable
        unsatisfied_bounds.sort();

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
    new_predicates: Option<&FxHashSet<ty::Clause<'tcx>>>,
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
    ty::ParamEnv::new(bounds, param_env.reveal())
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
) -> Option<FxHashSet<ty::Clause<'tcx>>> {
    // The bounds we that we would require from `to_check`
    let mut bounds = FxHashSet::default();

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
        if let ty::ReStatic | ty::ReError(_) = **region_a {
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
                    ty::EarlyParamRegion {
                        def_id: region_param.def_id,
                        index: region_param.index,
                        name: region_param.name,
                    },
                );
                // The predicate we expect to see. (In our example,
                // `Self: 'me`.)
                bounds.insert(
                    ty::ClauseKind::TypeOutlives(ty::OutlivesPredicate(ty_param, region_param))
                        .to_predicate(tcx),
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
            if matches!(**region_b, ty::ReStatic | ty::ReError(_)) || region_a == region_b {
                continue;
            }
            if region_known_to_outlive(tcx, item_def_id, param_env, wf_tys, *region_a, *region_b) {
                debug!(?region_a_idx, ?region_b_idx);
                debug!("required clause: {region_a} must outlive {region_b}");
                // Translate into the generic parameters of the GAT.
                let region_a_param = gat_generics.param_at(*region_a_idx, tcx);
                let region_a_param = ty::Region::new_early_param(
                    tcx,
                    ty::EarlyParamRegion {
                        def_id: region_a_param.def_id,
                        index: region_a_param.index,
                        name: region_a_param.name,
                    },
                );
                // Same for the region.
                let region_b_param = gat_generics.param_at(*region_b_idx, tcx);
                let region_b_param = ty::Region::new_early_param(
                    tcx,
                    ty::EarlyParamRegion {
                        def_id: region_b_param.def_id,
                        index: region_b_param.index,
                        name: region_b_param.name,
                    },
                );
                // The predicate we expect to see.
                bounds.insert(
                    ty::ClauseKind::RegionOutlives(ty::OutlivesPredicate(
                        region_a_param,
                        region_b_param,
                    ))
                    .to_predicate(tcx),
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
        infcx.register_region_obligation(infer::RegionObligation {
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
        infcx.sub_regions(infer::RelateRegionParamBound(DUMMY_SP), region_b, region_a);
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
    let infcx = tcx.infer_ctxt().build();

    add_constraints(&infcx);

    let outlives_environment = OutlivesEnvironment::with_bounds(
        param_env,
        infcx.implied_bounds_tys(param_env, id, wf_tys),
    );

    let errors = infcx.resolve_regions(&outlives_environment);
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
    regions: FxHashSet<(ty::Region<'tcx>, usize)>,
    // Which params appears and which parameter index its instantiated with
    types: FxHashSet<(Ty<'tcx>, usize)>,
}

impl<'tcx> GATArgsCollector<'tcx> {
    fn visit<T: TypeFoldable<TyCtxt<'tcx>>>(
        gat: DefId,
        t: T,
    ) -> (FxHashSet<(ty::Region<'tcx>, usize)>, FxHashSet<(Ty<'tcx>, usize)>) {
        let mut visitor =
            GATArgsCollector { gat, regions: FxHashSet::default(), types: FxHashSet::default() };
        t.visit_with(&mut visitor);
        (visitor.regions, visitor.types)
    }
}

impl<'tcx> TypeVisitor<TyCtxt<'tcx>> for GATArgsCollector<'tcx> {
    type BreakTy = !;

    fn visit_ty(&mut self, t: Ty<'tcx>) -> ControlFlow<Self::BreakTy> {
        match t.kind() {
            ty::Alias(ty::Projection, p) if p.def_id == self.gat => {
                for (idx, arg) in p.args.iter().enumerate() {
                    match arg.unpack() {
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

/// Detect when an object unsafe trait is referring to itself in one of its associated items.
/// When this is done, suggest using `Self` instead.
fn check_object_unsafe_self_trait_by_name(tcx: TyCtxt<'_>, item: &hir::TraitItem<'_>) {
    let (trait_name, trait_def_id) =
        match tcx.hir_node_by_def_id(tcx.hir().get_parent_item(item.hir_id()).def_id) {
            hir::Node::Item(item) => match item.kind {
                hir::ItemKind::Trait(..) => (item.ident, item.owner_id),
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
        if tcx.check_is_object_safe(trait_def_id) {
            return;
        }
        let sugg = trait_should_be_self.iter().map(|span| (*span, "Self".to_string())).collect();
        tcx.dcx()
            .struct_span_err(
                trait_should_be_self,
                "associated item referring to unboxed trait object for its own trait",
            )
            .with_span_label(trait_name.span, "in this trait")
            .with_multipart_suggestion(
                "you might have meant to use `Self` to refer to the implementing type",
                sugg,
                Applicability::MachineApplicable,
            )
            .emit();
    }
}

fn check_impl_item(tcx: TyCtxt<'_>, impl_item: &hir::ImplItem<'_>) -> Result<(), ErrorGuaranteed> {
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
        hir::GenericParamKind::Const { ty: hir_ty, default: _, is_host_effect: _ } => {
            let ty = tcx.type_of(param.def_id).instantiate_identity();

            if tcx.features().adt_const_params {
                enter_wf_checking_ctxt(tcx, hir_ty.span, param.def_id, |wfcx| {
                    let trait_def_id =
                        tcx.require_lang_item(LangItem::ConstParamTy, Some(hir_ty.span));
                    wfcx.register_bound(
                        ObligationCause::new(
                            hir_ty.span,
                            param.def_id,
                            ObligationCauseCode::ConstParam(ty),
                        ),
                        wfcx.param_env,
                        ty,
                        trait_def_id,
                    );
                    Ok(())
                })
            } else {
                let mut diag = match ty.kind() {
                    ty::Bool | ty::Char | ty::Int(_) | ty::Uint(_) | ty::Error(_) => return Ok(()),
                    ty::FnPtr(_) => tcx.dcx().struct_span_err(
                        hir_ty.span,
                        "using function pointers as const generic parameters is forbidden",
                    ),
                    ty::RawPtr(_) => tcx.dcx().struct_span_err(
                        hir_ty.span,
                        "using raw pointers as const generic parameters is forbidden",
                    ),
                    _ => tcx.dcx().struct_span_err(
                        hir_ty.span,
                        format!("`{}` is forbidden as the type of a const generic parameter", ty),
                    ),
                };

                diag.note("the only supported types are integers, `bool` and `char`");

                let cause = ObligationCause::misc(hir_ty.span, param.def_id);
                let may_suggest_feature = match type_allowed_to_implement_const_param_ty(
                    tcx,
                    tcx.param_env(param.def_id),
                    ty,
                    cause,
                ) {
                    // Can never implement `ConstParamTy`, don't suggest anything.
                    Err(ConstParamTyImplementationError::NotAnAdtOrBuiltinAllowed) => false,
                    // May be able to implement `ConstParamTy`. Only emit the feature help
                    // if the type is local, since the user may be able to fix the local type.
                    Err(ConstParamTyImplementationError::InfrigingFields(..)) => {
                        fn ty_is_local(ty: Ty<'_>) -> bool {
                            match ty.kind() {
                                ty::Adt(adt_def, ..) => adt_def.did().is_local(),
                                // Arrays and slices use the inner type's `ConstParamTy`.
                                ty::Array(ty, ..) => ty_is_local(*ty),
                                ty::Slice(ty) => ty_is_local(*ty),
                                // `&` references use the inner type's `ConstParamTy`.
                                // `&mut` are not supported.
                                ty::Ref(_, ty, ast::Mutability::Not) => ty_is_local(*ty),
                                // Say that a tuple is local if any of its components are local.
                                // This is not strictly correct, but it's likely that the user can fix the local component.
                                ty::Tuple(tys) => tys.iter().any(|ty| ty_is_local(ty)),
                                _ => false,
                            }
                        }

                        ty_is_local(ty)
                    }
                    // Implments `ConstParamTy`, suggest adding the feature to enable.
                    Ok(..) => true,
                };
                if may_suggest_feature && tcx.sess.is_nightly_build() {
                    diag.help(
                        "add `#![feature(adt_const_params)]` to the crate attributes to enable more complex and user defined types",
                    );
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
        tcx.ensure()
            .coherent_trait(tcx.parent(item.trait_item_def_id.unwrap_or(item_id.into())))?;

        let self_ty = match item.container {
            ty::TraitContainer => tcx.types.self_param,
            ty::ImplContainer => tcx.type_of(item.container_id(tcx)).instantiate_identity(),
        };

        match item.kind {
            ty::AssocKind::Const => {
                let ty = tcx.type_of(item.def_id).instantiate_identity();
                let ty = wfcx.normalize(span, Some(WellFormedLoc::Ty(item_id)), ty);
                wfcx.register_wf_obligation(span, loc, ty.into());
                Ok(())
            }
            ty::AssocKind::Fn => {
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
            ty::AssocKind::Type => {
                if let ty::AssocItemContainer::TraitContainer = item.container {
                    check_associated_type_bounds(wfcx, item, span)
                }
                if item.defaultness(tcx).has_value() {
                    let ty = tcx.type_of(item.def_id).instantiate_identity();
                    let ty = wfcx.normalize(span, Some(WellFormedLoc::Ty(item_id)), ty);
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
                let field_id = field.did.expect_local();
                let hir::FieldDef { ty: hir_ty, .. } =
                    tcx.hir_node_by_def_id(field_id).expect_field();
                let ty = wfcx.normalize(
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
                    if ty.has_infer() {
                        tcx.dcx()
                            .span_delayed_bug(item.span, format!("inference variables in {ty:?}"));
                        // Just treat unresolved type expression as if it needs drop.
                        true
                    } else {
                        ty.needs_drop(tcx, tcx.param_env(item.owner_id))
                    }
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
                        traits::FieldSized {
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
                    tcx.require_lang_item(LangItem::Sized, None),
                );
            }

            // Explicit `enum` discriminant values must const-evaluate successfully.
            if let ty::VariantDiscr::Explicit(discr_def_id) = variant.discr {
                let cause = traits::ObligationCause::new(
                    tcx.def_span(discr_def_id),
                    wfcx.body_def_id,
                    traits::MiscObligation,
                );
                wfcx.register_obligation(traits::Obligation::new(
                    tcx,
                    cause,
                    wfcx.param_env,
                    ty::Binder::dummy(ty::PredicateKind::Clause(ty::ClauseKind::ConstEvaluatable(
                        ty::Const::from_anon_const(tcx, discr_def_id.expect_local()),
                    ))),
                ));
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
    let wf_obligations =
        bounds.instantiate_identity_iter_copied().flat_map(|(bound, bound_span)| {
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
    Allow,
    AllowIfForeignTail,
}

fn check_item_type(
    tcx: TyCtxt<'_>,
    item_id: LocalDefId,
    ty_span: Span,
    unsized_handling: UnsizedHandling,
) -> Result<(), ErrorGuaranteed> {
    debug!("check_item_type: {:?}", item_id);

    enter_wf_checking_ctxt(tcx, ty_span, item_id, |wfcx| {
        let ty = tcx.type_of(item_id).instantiate_identity();
        let item_ty = wfcx.normalize(ty_span, Some(WellFormedLoc::Ty(item_id)), ty);

        let forbid_unsized = match unsized_handling {
            UnsizedHandling::Forbid => true,
            UnsizedHandling::Allow => false,
            UnsizedHandling::AllowIfForeignTail => {
                let tail = tcx.struct_tail_erasing_lifetimes(item_ty, wfcx.param_env);
                !matches!(tail.kind(), ty::Foreign(_))
            }
        };

        wfcx.register_wf_obligation(ty_span, Some(WellFormedLoc::Ty(item_id)), item_ty.into());
        if forbid_unsized {
            wfcx.register_bound(
                traits::ObligationCause::new(ty_span, wfcx.body_def_id, traits::WellFormed(None)),
                wfcx.param_env,
                item_ty,
                tcx.require_lang_item(LangItem::Sized, None),
            );
        }

        // Ensure that the end result is `Sync` in a non-thread local `static`.
        let should_check_for_sync = tcx.static_mutability(item_id.to_def_id())
            == Some(hir::Mutability::Not)
            && !tcx.is_foreign_item(item_id.to_def_id())
            && !tcx.is_thread_local_static(item_id.to_def_id());

        if should_check_for_sync {
            wfcx.register_bound(
                traits::ObligationCause::new(ty_span, wfcx.body_def_id, traits::SharedStatic),
                wfcx.param_env,
                item_ty,
                tcx.require_lang_item(LangItem::Sync, Some(ty_span)),
            );
        }
        Ok(())
    })
}

#[instrument(level = "debug", skip(tcx, ast_self_ty, ast_trait_ref))]
fn check_impl<'tcx>(
    tcx: TyCtxt<'tcx>,
    item: &'tcx hir::Item<'tcx>,
    ast_self_ty: &hir::Ty<'_>,
    ast_trait_ref: &Option<hir::TraitRef<'_>>,
) -> Result<(), ErrorGuaranteed> {
    enter_wf_checking_ctxt(tcx, item.span, item.owner_id.def_id, |wfcx| {
        match ast_trait_ref {
            Some(ast_trait_ref) => {
                // `#[rustc_reservation_impl]` impls are not real impls and
                // therefore don't need to be WF (the trait's `Self: Trait` predicate
                // won't hold).
                let trait_ref = tcx.impl_trait_ref(item.owner_id).unwrap().instantiate_identity();
                // Avoid bogus "type annotations needed `Foo: Bar`" errors on `impl Bar for Foo` in case
                // other `Foo` impls are incoherent.
                tcx.ensure().coherent_trait(trait_ref.def_id)?;
                let trait_ref = wfcx.normalize(
                    ast_trait_ref.path.span,
                    Some(WellFormedLoc::Ty(item.hir_id().expect_owner().def_id)),
                    trait_ref,
                );
                let trait_pred =
                    ty::TraitPredicate { trait_ref, polarity: ty::ImplPolarity::Positive };
                let mut obligations = traits::wf::trait_obligations(
                    wfcx.infcx,
                    wfcx.param_env,
                    wfcx.body_def_id,
                    trait_pred,
                    ast_trait_ref.path.span,
                    item,
                );
                for obligation in &mut obligations {
                    if let Some(pred) = obligation.predicate.to_opt_poly_trait_pred()
                        && pred.self_ty().skip_binder() == trait_ref.self_ty()
                    {
                        obligation.cause.span = ast_self_ty.span;
                    }
                }
                debug!(?obligations);
                wfcx.register_obligations(obligations);
            }
            None => {
                let self_ty = tcx.type_of(item.owner_id).instantiate_identity();
                let self_ty = wfcx.normalize(
                    item.span,
                    Some(WellFormedLoc::Ty(item.hir_id().expect_owner().def_id)),
                    self_ty,
                );
                wfcx.register_wf_obligation(
                    ast_self_ty.span,
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

    let is_our_default = |def: &ty::GenericParamDef| match def.kind {
        GenericParamDefKind::Type { has_default, .. }
        | GenericParamDefKind::Const { has_default, .. } => {
            has_default && def.index >= generics.parent_count as u32
        }
        GenericParamDefKind::Lifetime => {
            span_bug!(tcx.def_span(def.def_id), "lifetime params can have no default")
        }
    };

    // Check that concrete defaults are well-formed. See test `type-check-defaults.rs`.
    // For example, this forbids the declaration:
    //
    //     struct Foo<T = Vec<[u32]>> { .. }
    //
    // Here, the default `Vec<[u32]>` is not WF because `[u32]: Sized` does not hold.
    for param in &generics.params {
        match param.kind {
            GenericParamDefKind::Type { .. } => {
                if is_our_default(param) {
                    let ty = tcx.type_of(param.def_id).instantiate_identity();
                    // Ignore dependent defaults -- that is, where the default of one type
                    // parameter includes another (e.g., `<T, U = T>`). In those cases, we can't
                    // be sure if it will error or not as user might always specify the other.
                    if !ty.has_param() {
                        wfcx.register_wf_obligation(
                            tcx.def_span(param.def_id),
                            Some(WellFormedLoc::Ty(param.def_id.expect_local())),
                            ty.into(),
                        );
                    }
                }
            }
            GenericParamDefKind::Const { .. } => {
                if is_our_default(param) {
                    // FIXME(const_generics_defaults): This
                    // is incorrect when dealing with unused args, for example
                    // for `struct Foo<const N: usize, const M: usize = { 1 - 2 }>`
                    // we should eagerly error.
                    let default_ct = tcx.const_param_default(param.def_id).instantiate_identity();
                    if !default_ct.has_param() {
                        wfcx.register_wf_obligation(
                            tcx.def_span(param.def_id),
                            None,
                            default_ct.into(),
                        );
                    }
                }
            }
            // Doesn't have defaults.
            GenericParamDefKind::Lifetime => {}
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
        match param.kind {
            GenericParamDefKind::Lifetime => {
                // All regions are identity.
                tcx.mk_param_from_def(param)
            }

            GenericParamDefKind::Type { .. } => {
                // If the param has a default, ...
                if is_our_default(param) {
                    let default_ty = tcx.type_of(param.def_id).instantiate_identity();
                    // ... and it's not a dependent default, ...
                    if !default_ty.has_param() {
                        // ... then instantiate it with the default.
                        return default_ty.into();
                    }
                }

                tcx.mk_param_from_def(param)
            }
            GenericParamDefKind::Const { .. } => {
                // If the param has a default, ...
                if is_our_default(param) {
                    let default_ct = tcx.const_param_default(param.def_id).instantiate_identity();
                    // ... and it's not a dependent default, ...
                    if !default_ct.has_param() {
                        // ... then instantiate it with the default.
                        return default_ct.into();
                    }
                }

                tcx.mk_param_from_def(param)
            }
        }
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
            impl<'tcx> ty::visit::TypeVisitor<TyCtxt<'tcx>> for CountParams {
                type BreakTy = ();

                fn visit_ty(&mut self, t: Ty<'tcx>) -> ControlFlow<Self::BreakTy> {
                    if let ty::Param(param) = t.kind() {
                        self.params.insert(param.index);
                    }
                    t.super_visit_with(self)
                }

                fn visit_region(&mut self, _: ty::Region<'tcx>) -> ControlFlow<Self::BreakTy> {
                    ControlFlow::Break(())
                }

                fn visit_const(&mut self, c: ty::Const<'tcx>) -> ControlFlow<Self::BreakTy> {
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
                traits::ItemObligation(def_id.to_def_id()),
            );
            traits::Obligation::new(tcx, cause, wfcx.param_env, pred)
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
            wfcx.normalize(
                arg_span(idx),
                Some(WellFormedLoc::Param {
                    function: def_id,
                    // Note that the `param_idx` of the output type is
                    // one greater than the index of the last input type.
                    param_idx: idx.try_into().unwrap(),
                }),
                ty,
            )
        }));

    for (idx, ty) in sig.inputs_and_output.iter().enumerate() {
        wfcx.register_wf_obligation(
            arg_span(idx),
            Some(WellFormedLoc::Param { function: def_id, param_idx: idx.try_into().unwrap() }),
            ty.into(),
        );
    }

    check_where_clauses(wfcx, span, def_id);

    if sig.abi == Abi::RustCall {
        let span = tcx.def_span(def_id);
        let has_implicit_self = hir_decl.implicit_self != hir::ImplicitSelfKind::None;
        let mut inputs = sig.inputs().iter().skip(if has_implicit_self { 1 } else { 0 });
        // Check that the argument is a tuple and is sized
        if let Some(ty) = inputs.next() {
            wfcx.register_bound(
                ObligationCause::new(span, wfcx.body_def_id, ObligationCauseCode::RustCall),
                wfcx.param_env,
                *ty,
                tcx.require_lang_item(hir::LangItem::Tuple, Some(span)),
            );
            wfcx.register_bound(
                ObligationCause::new(span, wfcx.body_def_id, ObligationCauseCode::RustCall),
                wfcx.param_env,
                *ty,
                tcx.require_lang_item(hir::LangItem::Sized, Some(span)),
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
}

const HELP_FOR_SELF_TYPE: &str = "consider changing to `self`, `&self`, `&mut self`, `self: Box<Self>`, \
     `self: Rc<Self>`, `self: Arc<Self>`, or `self: Pin<P>` (where P is one \
     of the previous types except `Self`)";

#[instrument(level = "debug", skip(wfcx))]
fn check_method_receiver<'tcx>(
    wfcx: &WfCheckingCtxt<'_, 'tcx>,
    fn_sig: &hir::FnSig<'_>,
    method: ty::AssocItem,
    self_ty: Ty<'tcx>,
) -> Result<(), ErrorGuaranteed> {
    let tcx = wfcx.tcx();

    if !method.fn_has_self_parameter {
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

    if tcx.features().arbitrary_self_types {
        if !receiver_is_valid(wfcx, span, receiver_ty, self_ty, true) {
            // Report error; `arbitrary_self_types` was enabled.
            return Err(e0307(tcx, span, receiver_ty));
        }
    } else {
        if !receiver_is_valid(wfcx, span, receiver_ty, self_ty, false) {
            return Err(if receiver_is_valid(wfcx, span, receiver_ty, self_ty, true) {
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
                .with_help(HELP_FOR_SELF_TYPE)
                .emit()
            } else {
                // Report error; would not have worked with `arbitrary_self_types`.
                e0307(tcx, span, receiver_ty)
            });
        }
    }
    Ok(())
}

fn e0307(tcx: TyCtxt<'_>, span: Span, receiver_ty: Ty<'_>) -> ErrorGuaranteed {
    struct_span_code_err!(tcx.dcx(), span, E0307, "invalid `self` parameter type: {receiver_ty}")
        .with_note("type of `self` must be `Self` or a type that dereferences to it")
        .with_help(HELP_FOR_SELF_TYPE)
        .emit()
}

/// Returns whether `receiver_ty` would be considered a valid receiver type for `self_ty`. If
/// `arbitrary_self_types` is enabled, `receiver_ty` must transitively deref to `self_ty`, possibly
/// through a `*const/mut T` raw pointer. If the feature is not enabled, the requirements are more
/// strict: `receiver_ty` must implement `Receiver` and directly implement
/// `Deref<Target = self_ty>`.
///
/// N.B., there are cases this function returns `true` but causes an error to be emitted,
/// particularly when `receiver_ty` derefs to a type that is the same as `self_ty` but has the
/// wrong lifetime. Be careful of this if you are calling this function speculatively.
fn receiver_is_valid<'tcx>(
    wfcx: &WfCheckingCtxt<'_, 'tcx>,
    span: Span,
    receiver_ty: Ty<'tcx>,
    self_ty: Ty<'tcx>,
    arbitrary_self_types_enabled: bool,
) -> bool {
    let infcx = wfcx.infcx;
    let tcx = wfcx.tcx();
    let cause =
        ObligationCause::new(span, wfcx.body_def_id, traits::ObligationCauseCode::MethodReceiver);

    let can_eq_self = |ty| infcx.can_eq(wfcx.param_env, self_ty, ty);

    // `self: Self` is always valid.
    if can_eq_self(receiver_ty) {
        if let Err(err) = wfcx.eq(&cause, wfcx.param_env, self_ty, receiver_ty) {
            infcx.err_ctxt().report_mismatched_types(&cause, self_ty, receiver_ty, err).emit();
        }
        return true;
    }

    let mut autoderef = Autoderef::new(infcx, wfcx.param_env, wfcx.body_def_id, span, receiver_ty);

    // The `arbitrary_self_types` feature allows raw pointer receivers like `self: *const Self`.
    if arbitrary_self_types_enabled {
        autoderef = autoderef.include_raw_pointers();
    }

    // The first type is `receiver_ty`, which we know its not equal to `self_ty`; skip it.
    autoderef.next();

    let receiver_trait_def_id = tcx.require_lang_item(LangItem::Receiver, Some(span));

    // Keep dereferencing `receiver_ty` until we get to `self_ty`.
    loop {
        if let Some((potential_self_ty, _)) = autoderef.next() {
            debug!(
                "receiver_is_valid: potential self type `{:?}` to match `{:?}`",
                potential_self_ty, self_ty
            );

            if can_eq_self(potential_self_ty) {
                wfcx.register_obligations(autoderef.into_obligations());

                if let Err(err) = wfcx.eq(&cause, wfcx.param_env, self_ty, potential_self_ty) {
                    infcx
                        .err_ctxt()
                        .report_mismatched_types(&cause, self_ty, potential_self_ty, err)
                        .emit();
                }

                break;
            } else {
                // Without `feature(arbitrary_self_types)`, we require that each step in the
                // deref chain implement `receiver`
                if !arbitrary_self_types_enabled
                    && !receiver_is_implemented(
                        wfcx,
                        receiver_trait_def_id,
                        cause.clone(),
                        potential_self_ty,
                    )
                {
                    return false;
                }
            }
        } else {
            debug!("receiver_is_valid: type `{:?}` does not deref to `{:?}`", receiver_ty, self_ty);
            return false;
        }
    }

    // Without `feature(arbitrary_self_types)`, we require that `receiver_ty` implements `Receiver`.
    if !arbitrary_self_types_enabled
        && !receiver_is_implemented(wfcx, receiver_trait_def_id, cause.clone(), receiver_ty)
    {
        return false;
    }

    true
}

fn receiver_is_implemented<'tcx>(
    wfcx: &WfCheckingCtxt<'_, 'tcx>,
    receiver_trait_def_id: DefId,
    cause: ObligationCause<'tcx>,
    receiver_ty: Ty<'tcx>,
) -> bool {
    let tcx = wfcx.tcx();
    let trait_ref = ty::TraitRef::new(tcx, receiver_trait_def_id, [receiver_ty]);

    let obligation = traits::Obligation::new(tcx, cause, wfcx.param_env, trait_ref);

    if wfcx.infcx.predicate_must_hold_modulo_regions(&obligation) {
        true
    } else {
        debug!(
            "receiver_is_implemented: type `{:?}` does not implement `Receiver` trait",
            receiver_ty
        );
        false
    }
}

fn check_variances_for_type_defn<'tcx>(
    tcx: TyCtxt<'tcx>,
    item: &hir::Item<'tcx>,
    hir_generics: &hir::Generics<'tcx>,
) {
    let identity_args = ty::GenericArgs::identity_for_item(tcx, item.owner_id);

    match item.kind {
        ItemKind::Enum(..) | ItemKind::Struct(..) | ItemKind::Union(..) => {
            for field in tcx.adt_def(item.owner_id).all_fields() {
                if field.ty(tcx, identity_args).references_error() {
                    return;
                }
            }
        }
        ItemKind::TyAlias(..) => {
            assert!(
                tcx.type_alias_is_lazy(item.owner_id),
                "should not be computing variance of non-weak type alias"
            );
            if tcx.type_of(item.owner_id).skip_binder().references_error() {
                return;
            }
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
            .filter_map(|predicate| match predicate {
                hir::WherePredicate::BoundPredicate(predicate) => {
                    match icx.to_ty(predicate.bounded_ty).kind() {
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

        let ty_param = &ty_generics.params[index];
        let hir_param = &hir_generics.params[index];

        if ty_param.def_id != hir_param.def_id.into() {
            // valid programs always have lifetimes before types in the generic parameter list
            // ty_generics are normalized to be in this required order, and variances are built
            // from ty generics, not from hir generics. but we need hir generics to get
            // a span out
            //
            // if they aren't in the same order, then the user has written invalid code, and already
            // got an error about it (or I'm wrong about this)
            tcx.dcx().span_delayed_bug(
                hir_param.span,
                "hir generics and ty generics in different order",
            );
            continue;
        }

        match hir_param.name {
            hir::ParamName::Error => {}
            _ => {
                let has_explicit_bounds = explicitly_bounded_params.contains(&parameter);
                report_bivariance(tcx, hir_param, has_explicit_bounds, item.kind);
            }
        }
    }
}

fn report_bivariance(
    tcx: TyCtxt<'_>,
    param: &rustc_hir::GenericParam<'_>,
    has_explicit_bounds: bool,
    item_kind: ItemKind<'_>,
) -> ErrorGuaranteed {
    let param_name = param.name.ident();

    let help = match item_kind {
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

    let const_param_help =
        matches!(param.kind, hir::GenericParamKind::Type { .. } if !has_explicit_bounds)
            .then_some(());

    let mut diag = tcx.dcx().create_err(errors::UnusedGenericParameter {
        span: param.span,
        param_name,
        param_def_kind: tcx.def_descr(param.def_id.to_def_id()),
        help,
        const_param_help,
    });
    diag.code(E0392);
    diag.emit()
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
                let hir_node = tcx.opt_hir_node_by_def_id(self.body_def_id);

                // only use the span of the predicate clause (#90869)

                if let Some(hir::Generics { predicates, .. }) =
                    hir_node.and_then(|node| node.generics())
                {
                    span = predicates
                        .iter()
                        // There seems to be no better way to find out which predicate we are in
                        .find(|pred| pred.span().contains(obligation_span))
                        .map(|pred| pred.span())
                        .unwrap_or(obligation_span);
                }

                let obligation = traits::Obligation::new(
                    tcx,
                    traits::ObligationCause::new(span, self.body_def_id, traits::TrivialBound),
                    empty_env,
                    pred,
                );
                self.ocx.register_obligation(obligation);
            }
        }
    }
}

fn check_mod_type_wf(tcx: TyCtxt<'_>, module: LocalModDefId) -> Result<(), ErrorGuaranteed> {
    let items = tcx.hir_module_items(module);
    let mut res = items.par_items(|item| tcx.ensure().check_well_formed(item.owner_id));
    res = res.and(items.par_impl_items(|item| tcx.ensure().check_well_formed(item.owner_id)));
    res = res.and(items.par_trait_items(|item| tcx.ensure().check_well_formed(item.owner_id)));
    res = res.and(items.par_foreign_items(|item| tcx.ensure().check_well_formed(item.owner_id)));
    if module == LocalModDefId::CRATE_DEF_ID {
        super::entry::check_for_entry_fn(tcx);
    }
    res
}

pub fn provide(providers: &mut Providers) {
    *providers = Providers { check_mod_type_wf, check_well_formed, ..*providers };
}
