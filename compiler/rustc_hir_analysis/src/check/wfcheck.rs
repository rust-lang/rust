use crate::autoderef::Autoderef;
use crate::constrained_generic_params::{identify_constrained_generic_params, Parameter};

use rustc_ast as ast;
use rustc_data_structures::fx::{FxHashMap, FxHashSet, FxIndexSet};
use rustc_errors::{pluralize, struct_span_err, Applicability, DiagnosticBuilder, ErrorGuaranteed};
use rustc_hir as hir;
use rustc_hir::def_id::{DefId, LocalDefId};
use rustc_hir::lang_items::LangItem;
use rustc_hir::ItemKind;
use rustc_infer::infer::outlives::env::{OutlivesEnvironment, RegionBoundPairs};
use rustc_infer::infer::outlives::obligations::TypeOutlives;
use rustc_infer::infer::{self, InferCtxt, TyCtxtInferExt};
use rustc_middle::mir::ConstraintCategory;
use rustc_middle::query::Providers;
use rustc_middle::ty::trait_def::TraitSpecializationKind;
use rustc_middle::ty::{
    self, AdtKind, GenericParamDefKind, Ty, TyCtxt, TypeFoldable, TypeSuperVisitable,
    TypeVisitable, TypeVisitableExt, TypeVisitor,
};
use rustc_middle::ty::{GenericArgKind, InternalSubsts};
use rustc_session::parse::feature_err;
use rustc_span::symbol::{sym, Ident, Symbol};
use rustc_span::{Span, DUMMY_SP};
use rustc_target::spec::abi::Abi;
use rustc_trait_selection::traits::error_reporting::TypeErrCtxtExt;
use rustc_trait_selection::traits::outlives_bounds::InferCtxtExt as _;
use rustc_trait_selection::traits::query::evaluate_obligation::InferCtxtExt as _;
use rustc_trait_selection::traits::{
    self, ObligationCause, ObligationCauseCode, ObligationCtxt, WellFormedLoc,
};

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
        // for a type to be WF, we do not need to check if const trait predicates satisfy.
        let param_env = self.param_env.without_const();
        self.ocx.register_obligation(traits::Obligation::new(
            self.tcx(),
            cause,
            param_env,
            ty::Binder::dummy(ty::PredicateKind::Clause(ty::ClauseKind::WellFormed(arg))),
        ));
    }
}

pub(super) fn enter_wf_checking_ctxt<'tcx, F>(
    tcx: TyCtxt<'tcx>,
    span: Span,
    body_def_id: LocalDefId,
    f: F,
) where
    F: for<'a> FnOnce(&WfCheckingCtxt<'a, 'tcx>),
{
    let param_env = tcx.param_env(body_def_id);
    let infcx = &tcx.infer_ctxt().build();
    let ocx = ObligationCtxt::new(infcx);

    let mut wfcx = WfCheckingCtxt { ocx, span, body_def_id, param_env };

    if !tcx.features().trivial_bounds {
        wfcx.check_false_global_bounds()
    }
    f(&mut wfcx);

    let assumed_wf_types = wfcx.ocx.assumed_wf_types(param_env, span, body_def_id);
    let implied_bounds = infcx.implied_bounds_tys(param_env, body_def_id, assumed_wf_types);

    let errors = wfcx.select_all_or_error();
    if !errors.is_empty() {
        infcx.err_ctxt().report_fulfillment_errors(&errors);
        return;
    }

    let outlives_env = OutlivesEnvironment::with_bounds(param_env, implied_bounds);

    let _ = wfcx.ocx.resolve_regions_and_report_errors(body_def_id, &outlives_env);
}

fn check_well_formed(tcx: TyCtxt<'_>, def_id: hir::OwnerId) {
    let node = tcx.hir().owner(def_id);
    match node {
        hir::OwnerNode::Crate(_) => {}
        hir::OwnerNode::Item(item) => check_item(tcx, item),
        hir::OwnerNode::TraitItem(item) => check_trait_item(tcx, item),
        hir::OwnerNode::ImplItem(item) => check_impl_item(tcx, item),
        hir::OwnerNode::ForeignItem(item) => check_foreign_item(tcx, item),
    }

    if let Some(generics) = node.generics() {
        for param in generics.params {
            check_param_wf(tcx, param)
        }
    }
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
fn check_item<'tcx>(tcx: TyCtxt<'tcx>, item: &'tcx hir::Item<'tcx>) {
    let def_id = item.owner_id.def_id;

    debug!(
        ?item.owner_id,
        item.name = ? tcx.def_path_str(def_id)
    );

    match item.kind {
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
            let is_auto = tcx
                .impl_trait_ref(def_id)
                .is_some_and(|trait_ref| tcx.trait_is_auto(trait_ref.skip_binder().def_id));
            if let (hir::Defaultness::Default { .. }, true) = (impl_.defaultness, is_auto) {
                let sp = impl_.of_trait.as_ref().map_or(item.span, |t| t.path.span);
                let mut err =
                    tcx.sess.struct_span_err(sp, "impls of auto traits cannot be default");
                err.span_labels(impl_.defaultness_span, "default because of this");
                err.span_label(sp, "auto trait");
                err.emit();
            }
            // We match on both `ty::ImplPolarity` and `ast::ImplPolarity` just to get the `!` span.
            match (tcx.impl_polarity(def_id), impl_.polarity) {
                (ty::ImplPolarity::Positive, _) => {
                    check_impl(tcx, item, impl_.self_ty, &impl_.of_trait, impl_.constness);
                }
                (ty::ImplPolarity::Negative, ast::ImplPolarity::Negative(span)) => {
                    // FIXME(#27579): what amount of WF checking do we need for neg impls?
                    if let hir::Defaultness::Default { .. } = impl_.defaultness {
                        let mut spans = vec![span];
                        spans.extend(impl_.defaultness_span);
                        struct_span_err!(
                            tcx.sess,
                            spans,
                            E0750,
                            "negative impls cannot be default impls"
                        )
                        .emit();
                    }
                }
                (ty::ImplPolarity::Reservation, _) => {
                    // FIXME: what amount of WF checking do we need for reservation impls?
                }
                _ => unreachable!(),
            }
        }
        hir::ItemKind::Fn(ref sig, ..) => {
            check_item_fn(tcx, def_id, item.ident, item.span, sig.decl);
        }
        hir::ItemKind::Static(ty, ..) => {
            check_item_type(tcx, def_id, ty.span, UnsizedHandling::Forbid);
        }
        hir::ItemKind::Const(ty, ..) => {
            check_item_type(tcx, def_id, ty.span, UnsizedHandling::Forbid);
        }
        hir::ItemKind::Struct(_, ast_generics) => {
            check_type_defn(tcx, item, false);
            check_variances_for_type_defn(tcx, item, ast_generics);
        }
        hir::ItemKind::Union(_, ast_generics) => {
            check_type_defn(tcx, item, true);
            check_variances_for_type_defn(tcx, item, ast_generics);
        }
        hir::ItemKind::Enum(_, ast_generics) => {
            check_type_defn(tcx, item, true);
            check_variances_for_type_defn(tcx, item, ast_generics);
        }
        hir::ItemKind::Trait(..) => {
            check_trait(tcx, item);
        }
        hir::ItemKind::TraitAlias(..) => {
            check_trait(tcx, item);
        }
        // `ForeignItem`s are handled separately.
        hir::ItemKind::ForeignMod { .. } => {}
        hir::ItemKind::TyAlias(hir_ty, ..) => {
            if tcx.type_of(item.owner_id.def_id).skip_binder().has_opaque_types() {
                // Bounds are respected for `type X = impl Trait` and `type X = (impl Trait, Y);`
                check_item_type(tcx, def_id, hir_ty.span, UnsizedHandling::Allow);
            }
        }
        _ => {}
    }
}

fn check_foreign_item(tcx: TyCtxt<'_>, item: &hir::ForeignItem<'_>) {
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
        hir::ForeignItemKind::Type => (),
    }
}

fn check_trait_item(tcx: TyCtxt<'_>, trait_item: &hir::TraitItem<'_>) {
    let def_id = trait_item.owner_id.def_id;

    let (method_sig, span) = match trait_item.kind {
        hir::TraitItemKind::Fn(ref sig, _) => (Some(sig), trait_item.span),
        hir::TraitItemKind::Type(_bounds, Some(ty)) => (None, ty.span),
        _ => (None, trait_item.span),
    };
    check_object_unsafe_self_trait_by_name(tcx, trait_item);
    check_associated_item(tcx, def_id, span, method_sig);
}

/// Require that the user writes where clauses on GATs for the implicit
/// outlives bounds involving trait parameters in trait functions and
/// lifetimes passed as GAT substs. See `self-outlives-lint` test.
///
/// We use the following trait as an example throughout this function:
/// ```rust,ignore (this code fails due to this lint)
/// trait IntoIter {
///     type Iter<'a>: Iterator<Item = Self::Item<'a>>;
///     type Item<'a>;
///     fn into_iter<'a>(&'a self) -> Self::Iter<'a>;
/// }
/// ```
fn check_gat_where_clauses(tcx: TyCtxt<'_>, associated_items: &[hir::TraitItemRef]) {
    // Associates every GAT's def_id to a list of possibly missing bounds detected by this lint.
    let mut required_bounds_by_item = FxHashMap::default();

    // Loop over all GATs together, because if this lint suggests adding a where-clause bound
    // to one GAT, it might then require us to an additional bound on another GAT.
    // In our `IntoIter` example, we discover a missing `Self: 'a` bound on `Iter<'a>`, which
    // then in a second loop adds a `Self: 'a` bound to `Item` due to the relationship between
    // those GATs.
    loop {
        let mut should_continue = false;
        for gat_item in associated_items {
            let gat_def_id = gat_item.id.owner_id;
            let gat_item = tcx.associated_item(gat_def_id);
            // If this item is not an assoc ty, or has no substs, then it's not a GAT
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
            let mut new_required_bounds: Option<FxHashSet<ty::Predicate<'_>>> = None;
            for item in associated_items {
                let item_def_id = item.id.owner_id;
                // Skip our own GAT, since it does not constrain itself at all.
                if item_def_id == gat_def_id {
                    continue;
                }

                let param_env = tcx.param_env(item_def_id);

                let item_required_bounds = match item.kind {
                    // In our example, this corresponds to `into_iter` method
                    hir::AssocItemKind::Fn { .. } => {
                        // For methods, we check the function signature's return type for any GATs
                        // to constrain. In the `into_iter` case, we see that the return type
                        // `Self::Iter<'a>` is a GAT we want to gather any potential missing bounds from.
                        let sig: ty::FnSig<'_> = tcx.liberate_late_bound_regions(
                            item_def_id.to_def_id(),
                            tcx.fn_sig(item_def_id).subst_identity(),
                        );
                        gather_gat_bounds(
                            tcx,
                            param_env,
                            item_def_id,
                            sig.inputs_and_output,
                            // We also assume that all of the function signature's parameter types
                            // are well formed.
                            &sig.inputs().iter().copied().collect(),
                            gat_def_id.def_id,
                            gat_generics,
                        )
                    }
                    // In our example, this corresponds to the `Iter` and `Item` associated types
                    hir::AssocItemKind::Type => {
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
                                .subst_identity_iter_copied()
                                .collect::<Vec<_>>(),
                            &FxIndexSet::default(),
                            gat_def_id.def_id,
                            gat_generics,
                        )
                    }
                    hir::AssocItemKind::Const => None,
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
        let gat_item_hir = tcx.hir().expect_trait_item(gat_def_id.def_id);
        debug!(?required_bounds);
        let param_env = tcx.param_env(gat_def_id);

        let mut unsatisfied_bounds: Vec<_> = required_bounds
            .into_iter()
            .filter(|clause| match clause.kind().skip_binder() {
                ty::PredicateKind::Clause(ty::ClauseKind::RegionOutlives(
                    ty::OutlivesPredicate(a, b),
                )) => !region_known_to_outlive(
                    tcx,
                    gat_def_id.def_id,
                    param_env,
                    &FxIndexSet::default(),
                    a,
                    b,
                ),
                ty::PredicateKind::Clause(ty::ClauseKind::TypeOutlives(ty::OutlivesPredicate(
                    a,
                    b,
                ))) => !ty_known_to_outlive(
                    tcx,
                    gat_def_id.def_id,
                    param_env,
                    &FxIndexSet::default(),
                    a,
                    b,
                ),
                _ => bug!("Unexpected PredicateKind"),
            })
            .map(|clause| clause.to_string())
            .collect();

        // We sort so that order is predictable
        unsatisfied_bounds.sort();

        if !unsatisfied_bounds.is_empty() {
            let plural = pluralize!(unsatisfied_bounds.len());
            let mut err = tcx.sess.struct_span_err(
                gat_item_hir.span,
                format!("missing required bound{} on `{}`", plural, gat_item_hir.ident),
            );

            let suggestion = format!(
                "{} {}",
                gat_item_hir.generics.add_where_or_trailing_comma(),
                unsatisfied_bounds.join(", "),
            );
            err.span_suggestion(
                gat_item_hir.generics.tail_span_for_predicate_suggestion(),
                format!("add the required where clause{plural}"),
                suggestion,
                Applicability::MachineApplicable,
            );

            let bound =
                if unsatisfied_bounds.len() > 1 { "these bounds are" } else { "this bound is" };
            err.note(format!(
                "{} currently required to ensure that impls have maximum flexibility",
                bound
            ));
            err.note(
                "we are soliciting feedback, see issue #87479 \
                 <https://github.com/rust-lang/rust/issues/87479> \
                 for more information",
            );

            err.emit();
        }
    }
}

/// Add a new set of predicates to the caller_bounds of an existing param_env.
fn augment_param_env<'tcx>(
    tcx: TyCtxt<'tcx>,
    param_env: ty::ParamEnv<'tcx>,
    new_predicates: Option<&FxHashSet<ty::Predicate<'tcx>>>,
) -> ty::ParamEnv<'tcx> {
    let Some(new_predicates) = new_predicates else {
        return param_env;
    };

    if new_predicates.is_empty() {
        return param_env;
    }

    let bounds = tcx.mk_predicates_from_iter(
        param_env.caller_bounds().iter().chain(new_predicates.iter().cloned()),
    );
    // FIXME(compiler-errors): Perhaps there is a case where we need to normalize this
    // i.e. traits::normalize_param_env_or_error
    ty::ParamEnv::new(bounds, param_env.reveal(), param_env.constness())
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
    item_def_id: hir::OwnerId,
    to_check: T,
    wf_tys: &FxIndexSet<Ty<'tcx>>,
    gat_def_id: LocalDefId,
    gat_generics: &'tcx ty::Generics,
) -> Option<FxHashSet<ty::Predicate<'tcx>>> {
    // The bounds we that we would require from `to_check`
    let mut bounds = FxHashSet::default();

    let (regions, types) = GATSubstCollector::visit(gat_def_id.to_def_id(), to_check);

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
        // clues
        if let ty::ReStatic = **region_a {
            continue;
        }
        // For each region argument (e.g., `'a` in our example), check for a
        // relationship to the type arguments (e.g., `Self`). If there is an
        // outlives relationship (`Self: 'a`), then we want to ensure that is
        // reflected in a where clause on the GAT itself.
        for (ty, ty_idx) in &types {
            // In our example, requires that `Self: 'a`
            if ty_known_to_outlive(tcx, item_def_id.def_id, param_env, &wf_tys, *ty, *region_a) {
                debug!(?ty_idx, ?region_a_idx);
                debug!("required clause: {ty} must outlive {region_a}");
                // Translate into the generic parameters of the GAT. In
                // our example, the type was `Self`, which will also be
                // `Self` in the GAT.
                let ty_param = gat_generics.param_at(*ty_idx, tcx);
                let ty_param = tcx.mk_ty_param(ty_param.index, ty_param.name);
                // Same for the region. In our example, 'a corresponds
                // to the 'me parameter.
                let region_param = gat_generics.param_at(*region_a_idx, tcx);
                let region_param = ty::Region::new_early_bound(
                    tcx,
                    ty::EarlyBoundRegion {
                        def_id: region_param.def_id,
                        index: region_param.index,
                        name: region_param.name,
                    },
                );
                // The predicate we expect to see. (In our example,
                // `Self: 'me`.)
                let clause = ty::PredicateKind::Clause(ty::ClauseKind::TypeOutlives(
                    ty::OutlivesPredicate(ty_param, region_param),
                ));
                let clause = tcx.mk_predicate(ty::Binder::dummy(clause));
                bounds.insert(clause);
            }
        }

        // For each region argument (e.g., `'a` in our example), also check for a
        // relationship to the other region arguments. If there is an outlives
        // relationship, then we want to ensure that is reflected in the where clause
        // on the GAT itself.
        for (region_b, region_b_idx) in &regions {
            // Again, skip `'static` because it outlives everything. Also, we trivially
            // know that a region outlives itself.
            if ty::ReStatic == **region_b || region_a == region_b {
                continue;
            }
            if region_known_to_outlive(
                tcx,
                item_def_id.def_id,
                param_env,
                &wf_tys,
                *region_a,
                *region_b,
            ) {
                debug!(?region_a_idx, ?region_b_idx);
                debug!("required clause: {region_a} must outlive {region_b}");
                // Translate into the generic parameters of the GAT.
                let region_a_param = gat_generics.param_at(*region_a_idx, tcx);
                let region_a_param = ty::Region::new_early_bound(
                    tcx,
                    ty::EarlyBoundRegion {
                        def_id: region_a_param.def_id,
                        index: region_a_param.index,
                        name: region_a_param.name,
                    },
                );
                // Same for the region.
                let region_b_param = gat_generics.param_at(*region_b_idx, tcx);
                let region_b_param = ty::Region::new_early_bound(
                    tcx,
                    ty::EarlyBoundRegion {
                        def_id: region_b_param.def_id,
                        index: region_b_param.index,
                        name: region_b_param.name,
                    },
                );
                // The predicate we expect to see.
                let clause = ty::PredicateKind::Clause(ty::ClauseKind::RegionOutlives(
                    ty::OutlivesPredicate(region_a_param, region_b_param),
                ));
                let clause = tcx.mk_predicate(ty::Binder::dummy(clause));
                bounds.insert(clause);
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
    resolve_regions_with_wf_tys(tcx, id, param_env, &wf_tys, |infcx, region_bound_pairs| {
        let origin = infer::RelateParamBound(DUMMY_SP, ty, None);
        let outlives = &mut TypeOutlives::new(infcx, tcx, region_bound_pairs, None, param_env);
        outlives.type_must_outlive(origin, ty, region, ConstraintCategory::BoringNoLocation);
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
    resolve_regions_with_wf_tys(tcx, id, param_env, &wf_tys, |mut infcx, _| {
        use rustc_infer::infer::outlives::obligations::TypeOutlivesDelegate;
        let origin = infer::RelateRegionParamBound(DUMMY_SP);
        // `region_a: region_b` -> `region_b <= region_a`
        infcx.push_sub_region_constraint(
            origin,
            region_b,
            region_a,
            ConstraintCategory::BoringNoLocation,
        );
    })
}

/// Given a known `param_env` and a set of well formed types, set up an
/// `InferCtxt`, call the passed function (to e.g. set up region constraints
/// to be tested), then resolve region and return errors
fn resolve_regions_with_wf_tys<'tcx>(
    tcx: TyCtxt<'tcx>,
    id: LocalDefId,
    param_env: ty::ParamEnv<'tcx>,
    wf_tys: &FxIndexSet<Ty<'tcx>>,
    add_constraints: impl for<'a> FnOnce(&'a InferCtxt<'tcx>, &'a RegionBoundPairs<'tcx>),
) -> bool {
    // Unfortunately, we have to use a new `InferCtxt` each call, because
    // region constraints get added and solved there and we need to test each
    // call individually.
    let infcx = tcx.infer_ctxt().build();
    let outlives_environment = OutlivesEnvironment::with_bounds(
        param_env,
        infcx.implied_bounds_tys(param_env, id, wf_tys.clone()),
    );
    let region_bound_pairs = outlives_environment.region_bound_pairs();

    add_constraints(&infcx, region_bound_pairs);

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
struct GATSubstCollector<'tcx> {
    gat: DefId,
    // Which region appears and which parameter index its substituted for
    regions: FxHashSet<(ty::Region<'tcx>, usize)>,
    // Which params appears and which parameter index its substituted for
    types: FxHashSet<(Ty<'tcx>, usize)>,
}

impl<'tcx> GATSubstCollector<'tcx> {
    fn visit<T: TypeFoldable<TyCtxt<'tcx>>>(
        gat: DefId,
        t: T,
    ) -> (FxHashSet<(ty::Region<'tcx>, usize)>, FxHashSet<(Ty<'tcx>, usize)>) {
        let mut visitor =
            GATSubstCollector { gat, regions: FxHashSet::default(), types: FxHashSet::default() };
        t.visit_with(&mut visitor);
        (visitor.regions, visitor.types)
    }
}

impl<'tcx> TypeVisitor<TyCtxt<'tcx>> for GATSubstCollector<'tcx> {
    type BreakTy = !;

    fn visit_ty(&mut self, t: Ty<'tcx>) -> ControlFlow<Self::BreakTy> {
        match t.kind() {
            ty::Alias(ty::Projection, p) if p.def_id == self.gat => {
                for (idx, subst) in p.substs.iter().enumerate() {
                    match subst.unpack() {
                        GenericArgKind::Lifetime(lt) if !lt.is_late_bound() => {
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
        match tcx.hir().get_by_def_id(tcx.hir().get_parent_item(item.hir_id()).def_id) {
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
        tcx.sess
            .struct_span_err(
                trait_should_be_self,
                "associated item referring to unboxed trait object for its own trait",
            )
            .span_label(trait_name.span, "in this trait")
            .multipart_suggestion(
                "you might have meant to use `Self` to refer to the implementing type",
                sugg,
                Applicability::MachineApplicable,
            )
            .emit();
    }
}

fn check_impl_item(tcx: TyCtxt<'_>, impl_item: &hir::ImplItem<'_>) {
    let (method_sig, span) = match impl_item.kind {
        hir::ImplItemKind::Fn(ref sig, _) => (Some(sig), impl_item.span),
        // Constrain binding and overflow error spans to `<Ty>` in `type foo = <Ty>`.
        hir::ImplItemKind::Type(ty) if ty.span != DUMMY_SP => (None, ty.span),
        _ => (None, impl_item.span),
    };

    check_associated_item(tcx, impl_item.owner_id.def_id, span, method_sig);
}

fn check_param_wf(tcx: TyCtxt<'_>, param: &hir::GenericParam<'_>) {
    match param.kind {
        // We currently only check wf of const params here.
        hir::GenericParamKind::Lifetime { .. } | hir::GenericParamKind::Type { .. } => (),

        // Const parameters are well formed if their type is structural match.
        hir::GenericParamKind::Const { ty: hir_ty, default: _ } => {
            let ty = tcx.type_of(param.def_id).subst_identity();

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
                });
            } else {
                let err_ty_str;
                let mut is_ptr = true;

                let err = match ty.kind() {
                    ty::Bool | ty::Char | ty::Int(_) | ty::Uint(_) | ty::Error(_) => None,
                    ty::FnPtr(_) => Some("function pointers"),
                    ty::RawPtr(_) => Some("raw pointers"),
                    _ => {
                        is_ptr = false;
                        err_ty_str = format!("`{ty}`");
                        Some(err_ty_str.as_str())
                    }
                };

                if let Some(unsupported_type) = err {
                    if is_ptr {
                        tcx.sess.span_err(
                            hir_ty.span,
                            format!(
                                "using {unsupported_type} as const generic parameters is forbidden",
                            ),
                        );
                    } else {
                        let mut err = tcx.sess.struct_span_err(
                            hir_ty.span,
                            format!(
                                "{unsupported_type} is forbidden as the type of a const generic parameter",
                            ),
                        );
                        err.note("the only supported types are integers, `bool` and `char`");
                        if tcx.sess.is_nightly_build() {
                            err.help(
                            "more complex types are supported with `#![feature(adt_const_params)]`",
                        );
                        }
                        err.emit();
                    }
                }
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
) {
    let loc = Some(WellFormedLoc::Ty(item_id));
    enter_wf_checking_ctxt(tcx, span, item_id, |wfcx| {
        let item = tcx.associated_item(item_id);

        let self_ty = match item.container {
            ty::TraitContainer => tcx.types.self_param,
            ty::ImplContainer => tcx.type_of(item.container_id(tcx)).subst_identity(),
        };

        match item.kind {
            ty::AssocKind::Const => {
                let ty = tcx.type_of(item.def_id).subst_identity();
                let ty = wfcx.normalize(span, Some(WellFormedLoc::Ty(item_id)), ty);
                wfcx.register_wf_obligation(span, loc, ty.into());
            }
            ty::AssocKind::Fn => {
                let sig = tcx.fn_sig(item.def_id).subst_identity();
                let hir_sig = sig_if_method.expect("bad signature for method");
                check_fn_or_method(
                    wfcx,
                    item.ident(tcx).span,
                    sig,
                    hir_sig.decl,
                    item.def_id.expect_local(),
                );
                check_method_receiver(wfcx, hir_sig, item, self_ty);
            }
            ty::AssocKind::Type => {
                if let ty::AssocItemContainer::TraitContainer = item.container {
                    check_associated_type_bounds(wfcx, item, span)
                }
                if item.defaultness(tcx).has_value() {
                    let ty = tcx.type_of(item.def_id).subst_identity();
                    let ty = wfcx.normalize(span, Some(WellFormedLoc::Ty(item_id)), ty);
                    wfcx.register_wf_obligation(span, loc, ty.into());
                }
            }
        }
    })
}

fn item_adt_kind(kind: &ItemKind<'_>) -> Option<AdtKind> {
    match kind {
        ItemKind::Struct(..) => Some(AdtKind::Struct),
        ItemKind::Union(..) => Some(AdtKind::Union),
        ItemKind::Enum(..) => Some(AdtKind::Enum),
        _ => None,
    }
}

/// In a type definition, we check that to ensure that the types of the fields are well-formed.
fn check_type_defn<'tcx>(tcx: TyCtxt<'tcx>, item: &hir::Item<'tcx>, all_sized: bool) {
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
                    tcx.hir().get_by_def_id(field_id).expect_field();
                let ty = wfcx.normalize(hir_ty.span, None, tcx.type_of(field.did).subst_identity());
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
                    let ty = tcx.type_of(variant.fields.raw.last().unwrap().did).subst_identity();
                    let ty = tcx.erase_regions(ty);
                    if ty.has_infer() {
                        tcx.sess
                            .delay_span_bug(item.span, format!("inference variables in {:?}", ty));
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
                    tcx.hir().get_by_def_id(field_id).expect_field();
                let ty = wfcx.normalize(hir_ty.span, None, tcx.type_of(field.did).subst_identity());
                wfcx.register_bound(
                    traits::ObligationCause::new(
                        hir_ty.span,
                        wfcx.body_def_id,
                        traits::FieldSized {
                            adt_kind: match item_adt_kind(&item.kind) {
                                Some(i) => i,
                                None => bug!(),
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
    });
}

#[instrument(skip(tcx, item))]
fn check_trait(tcx: TyCtxt<'_>, item: &hir::Item<'_>) {
    debug!(?item.owner_id);

    let def_id = item.owner_id.def_id;
    let trait_def = tcx.trait_def(def_id);
    if trait_def.is_marker
        || matches!(trait_def.specialization_kind, TraitSpecializationKind::Marker)
    {
        for associated_def_id in &*tcx.associated_item_def_ids(def_id) {
            struct_span_err!(
                tcx.sess,
                tcx.def_span(*associated_def_id),
                E0714,
                "marker traits cannot have associated items",
            )
            .emit();
        }
    }

    enter_wf_checking_ctxt(tcx, item.span, def_id, |wfcx| {
        check_where_clauses(wfcx, item.span, def_id)
    });

    // Only check traits, don't check trait aliases
    if let hir::ItemKind::Trait(_, _, _, _, items) = item.kind {
        check_gat_where_clauses(tcx, items);
    }
}

/// Checks all associated type defaults of trait `trait_def_id`.
///
/// Assuming the defaults are used, check that all predicates (bounds on the
/// assoc type and where clauses on the trait) hold.
fn check_associated_type_bounds(wfcx: &WfCheckingCtxt<'_, '_>, item: ty::AssocItem, span: Span) {
    let bounds = wfcx.tcx().explicit_item_bounds(item.def_id);

    debug!("check_associated_type_bounds: bounds={:?}", bounds);
    let wf_obligations = bounds.subst_identity_iter_copied().flat_map(|(bound, bound_span)| {
        let normalized_bound = wfcx.normalize(span, None, bound);
        traits::wf::predicate_obligations(
            wfcx.infcx,
            wfcx.param_env,
            wfcx.body_def_id,
            normalized_bound.as_predicate(),
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
) {
    enter_wf_checking_ctxt(tcx, span, def_id, |wfcx| {
        let sig = tcx.fn_sig(def_id).subst_identity();
        check_fn_or_method(wfcx, ident.span, sig, decl, def_id);
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
) {
    debug!("check_item_type: {:?}", item_id);

    enter_wf_checking_ctxt(tcx, ty_span, item_id, |wfcx| {
        let ty = tcx.type_of(item_id).subst_identity();
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
    });
}

#[instrument(level = "debug", skip(tcx, ast_self_ty, ast_trait_ref))]
fn check_impl<'tcx>(
    tcx: TyCtxt<'tcx>,
    item: &'tcx hir::Item<'tcx>,
    ast_self_ty: &hir::Ty<'_>,
    ast_trait_ref: &Option<hir::TraitRef<'_>>,
    constness: hir::Constness,
) {
    enter_wf_checking_ctxt(tcx, item.span, item.owner_id.def_id, |wfcx| {
        match ast_trait_ref {
            Some(ast_trait_ref) => {
                // `#[rustc_reservation_impl]` impls are not real impls and
                // therefore don't need to be WF (the trait's `Self: Trait` predicate
                // won't hold).
                let trait_ref = tcx.impl_trait_ref(item.owner_id).unwrap().subst_identity();
                let trait_ref = wfcx.normalize(
                    ast_trait_ref.path.span,
                    Some(WellFormedLoc::Ty(item.hir_id().expect_owner().def_id)),
                    trait_ref,
                );
                let trait_pred = ty::TraitPredicate {
                    trait_ref,
                    constness: match constness {
                        hir::Constness::Const => ty::BoundConstness::ConstIfConst,
                        hir::Constness::NotConst => ty::BoundConstness::NotConst,
                    },
                    polarity: ty::ImplPolarity::Positive,
                };
                let mut obligations = traits::wf::trait_obligations(
                    wfcx.infcx,
                    wfcx.param_env,
                    wfcx.body_def_id,
                    &trait_pred,
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
                let self_ty = tcx.type_of(item.owner_id).subst_identity();
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
    });
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
        | GenericParamDefKind::Const { has_default } => {
            has_default && def.index >= generics.parent_count as u32
        }
        GenericParamDefKind::Lifetime => unreachable!(),
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
                    let ty = tcx.type_of(param.def_id).subst_identity();
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
                    // is incorrect when dealing with unused substs, for example
                    // for `struct Foo<const N: usize, const M: usize = { 1 - 2 }>`
                    // we should eagerly error.
                    let default_ct = tcx.const_param_default(param.def_id).subst_identity();
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

    // Check that trait predicates are WF when params are substituted by their defaults.
    // We don't want to overly constrain the predicates that may be written but we want to
    // catch cases where a default my never be applied such as `struct Foo<T: Copy = String>`.
    // Therefore we check if a predicate which contains a single type param
    // with a concrete default is WF with that default substituted.
    // For more examples see tests `defaults-well-formedness.rs` and `type-check-defaults.rs`.
    //
    // First we build the defaulted substitution.
    let substs = InternalSubsts::for_item(tcx, def_id.to_def_id(), |param, _| {
        match param.kind {
            GenericParamDefKind::Lifetime => {
                // All regions are identity.
                tcx.mk_param_from_def(param)
            }

            GenericParamDefKind::Type { .. } => {
                // If the param has a default, ...
                if is_our_default(param) {
                    let default_ty = tcx.type_of(param.def_id).subst_identity();
                    // ... and it's not a dependent default, ...
                    if !default_ty.has_param() {
                        // ... then substitute it with the default.
                        return default_ty.into();
                    }
                }

                tcx.mk_param_from_def(param)
            }
            GenericParamDefKind::Const { .. } => {
                // If the param has a default, ...
                if is_our_default(param) {
                    let default_ct = tcx.const_param_default(param.def_id).subst_identity();
                    // ... and it's not a dependent default, ...
                    if !default_ct.has_param() {
                        // ... then substitute it with the default.
                        return default_ct.into();
                    }
                }

                tcx.mk_param_from_def(param)
            }
        }
    });

    // Now we build the substituted predicates.
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
            let substituted_pred = ty::EarlyBinder::bind(pred).subst(tcx, substs);
            // Don't check non-defaulted params, dependent defaults (including lifetimes)
            // or preds with multiple params.
            if substituted_pred.has_non_region_param() || param_count.params.len() > 1 || has_region
            {
                None
            } else if predicates.predicates.iter().any(|&(p, _)| p == substituted_pred) {
                // Avoid duplication of predicates that contain no parameters, for example.
                None
            } else {
                Some((substituted_pred, sp))
            }
        })
        .map(|(pred, sp)| {
            // Convert each of those into an obligation. So if you have
            // something like `struct Foo<T: Copy = String>`, we would
            // take that predicate `T: Copy`, substitute to `String: Copy`
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
        traits::wf::predicate_obligations(
            infcx,
            wfcx.param_env.without_const(),
            wfcx.body_def_id,
            p,
            sp,
        )
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

    check_return_position_impl_trait_in_trait_bounds(
        wfcx,
        def_id,
        sig.output(),
        hir_decl.output.span(),
    );

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
            tcx.sess.span_err(
                hir_decl.inputs.last().map_or(span, |input| input.span),
                "functions with the \"rust-call\" ABI must take a single non-self tuple argument",
            );
        }
        // No more inputs other than the `self` type and the tuple type
        if inputs.next().is_some() {
            tcx.sess.span_err(
                hir_decl.inputs.last().map_or(span, |input| input.span),
                "functions with the \"rust-call\" ABI must take a single non-self tuple argument",
            );
        }
    }
}

/// Basically `check_associated_type_bounds`, but separated for now and should be
/// deduplicated when RPITITs get lowered into real associated items.
#[tracing::instrument(level = "trace", skip(wfcx))]
fn check_return_position_impl_trait_in_trait_bounds<'tcx>(
    wfcx: &WfCheckingCtxt<'_, 'tcx>,
    fn_def_id: LocalDefId,
    fn_output: Ty<'tcx>,
    span: Span,
) {
    let tcx = wfcx.tcx();
    let Some(assoc_item) = tcx.opt_associated_item(fn_def_id.to_def_id()) else {
        return;
    };
    if assoc_item.container != ty::AssocItemContainer::TraitContainer {
        return;
    }
    fn_output.visit_with(&mut ImplTraitInTraitFinder {
        wfcx,
        fn_def_id,
        depth: ty::INNERMOST,
        seen: FxHashSet::default(),
    });
}

// FIXME(-Zlower-impl-trait-in-trait-to-assoc-ty): Even with the new lowering
// strategy, we can't just call `check_associated_item` on the new RPITITs,
// because tests like `tests/ui/async-await/in-trait/implied-bounds.rs` will fail.
// That's because we need to check that the bounds of the RPITIT hold using
// the special substs that we create during opaque type lowering, otherwise we're
// getting a bunch of early bound and free regions mixed up... Haven't looked too
// deep into this, though.
struct ImplTraitInTraitFinder<'a, 'tcx> {
    wfcx: &'a WfCheckingCtxt<'a, 'tcx>,
    fn_def_id: LocalDefId,
    depth: ty::DebruijnIndex,
    seen: FxHashSet<DefId>,
}
impl<'tcx> TypeVisitor<TyCtxt<'tcx>> for ImplTraitInTraitFinder<'_, 'tcx> {
    type BreakTy = !;

    fn visit_ty(&mut self, ty: Ty<'tcx>) -> ControlFlow<!> {
        let tcx = self.wfcx.tcx();
        if let ty::Alias(ty::Opaque, unshifted_opaque_ty) = *ty.kind()
            && self.seen.insert(unshifted_opaque_ty.def_id)
            && let Some(opaque_def_id) = unshifted_opaque_ty.def_id.as_local()
            && let origin = tcx.opaque_type_origin(opaque_def_id)
            && let hir::OpaqueTyOrigin::FnReturn(source) | hir::OpaqueTyOrigin::AsyncFn(source) = origin
            && source == self.fn_def_id
        {
            let opaque_ty = tcx.fold_regions(unshifted_opaque_ty, |re, _depth| {
                match re.kind() {
                    ty::ReEarlyBound(_) | ty::ReFree(_) | ty::ReError(_) => re,
                    r => bug!("unexpected region: {r:?}"),
                }
            });
            for (bound, bound_span) in tcx
                .explicit_item_bounds(opaque_ty.def_id)
                .subst_iter_copied(tcx, opaque_ty.substs)
            {
                let bound = self.wfcx.normalize(bound_span, None, bound);
                self.wfcx.register_obligations(traits::wf::predicate_obligations(
                    self.wfcx.infcx,
                    self.wfcx.param_env,
                    self.wfcx.body_def_id,
                    bound.as_predicate(),
                    bound_span,
                ));
                // Set the debruijn index back to innermost here, since we already eagerly
                // shifted the substs that we use to generate these bounds. This is unfortunately
                // subtly different behavior than the `ImplTraitInTraitFinder` we use in `param_env`,
                // but that function doesn't actually need to normalize the bound it's visiting
                // (whereas we have to do so here)...
                let old_depth = std::mem::replace(&mut self.depth, ty::INNERMOST);
                bound.visit_with(self);
                self.depth = old_depth;
            }
        }
        ty.super_visit_with(self)
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
) {
    let tcx = wfcx.tcx();

    if !method.fn_has_self_parameter {
        return;
    }

    let span = fn_sig.decl.inputs[0].span;

    let sig = tcx.fn_sig(method.def_id).subst_identity();
    let sig = tcx.liberate_late_bound_regions(method.def_id, sig);
    let sig = wfcx.normalize(span, None, sig);

    debug!("check_method_receiver: sig={:?}", sig);

    let self_ty = wfcx.normalize(span, None, self_ty);

    let receiver_ty = sig.inputs()[0];
    let receiver_ty = wfcx.normalize(span, None, receiver_ty);

    if tcx.features().arbitrary_self_types {
        if !receiver_is_valid(wfcx, span, receiver_ty, self_ty, true) {
            // Report error; `arbitrary_self_types` was enabled.
            e0307(tcx, span, receiver_ty);
        }
    } else {
        if !receiver_is_valid(wfcx, span, receiver_ty, self_ty, false) {
            if receiver_is_valid(wfcx, span, receiver_ty, self_ty, true) {
                // Report error; would have worked with `arbitrary_self_types`.
                feature_err(
                    &tcx.sess.parse_sess,
                    sym::arbitrary_self_types,
                    span,
                    format!(
                        "`{receiver_ty}` cannot be used as the type of `self` without \
                         the `arbitrary_self_types` feature",
                    ),
                )
                .help(HELP_FOR_SELF_TYPE)
                .emit();
            } else {
                // Report error; would not have worked with `arbitrary_self_types`.
                e0307(tcx, span, receiver_ty);
            }
        }
    }
}

fn e0307(tcx: TyCtxt<'_>, span: Span, receiver_ty: Ty<'_>) {
    struct_span_err!(
        tcx.sess.diagnostic(),
        span,
        E0307,
        "invalid `self` parameter type: {receiver_ty}"
    )
    .note("type of `self` must be `Self` or a type that dereferences to it")
    .help(HELP_FOR_SELF_TYPE)
    .emit();
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
            // If the receiver already has errors reported due to it, consider it valid to avoid
            // unnecessary errors (#58712).
            return receiver_ty.references_error();
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
    hir_generics: &hir::Generics<'_>,
) {
    let identity_substs = ty::InternalSubsts::identity_for_item(tcx, item.owner_id);
    for field in tcx.adt_def(item.owner_id).all_fields() {
        if field.ty(tcx, identity_substs).references_error() {
            return;
        }
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

    for (index, _) in variances.iter().enumerate() {
        let parameter = Parameter(index as u32);

        if constrained_parameters.contains(&parameter) {
            continue;
        }

        let param = &hir_generics.params[index];

        match param.name {
            hir::ParamName::Error => {}
            _ => {
                let has_explicit_bounds = explicitly_bounded_params.contains(&parameter);
                report_bivariance(tcx, param, has_explicit_bounds);
            }
        }
    }
}

fn report_bivariance(
    tcx: TyCtxt<'_>,
    param: &rustc_hir::GenericParam<'_>,
    has_explicit_bounds: bool,
) -> ErrorGuaranteed {
    let span = param.span;
    let param_name = param.name.ident().name;
    let mut err = error_392(tcx, span, param_name);

    let suggested_marker_id = tcx.lang_items().phantom_data();
    // Help is available only in presence of lang items.
    let msg = if let Some(def_id) = suggested_marker_id {
        format!(
            "consider removing `{}`, referring to it in a field, or using a marker such as `{}`",
            param_name,
            tcx.def_path_str(def_id),
        )
    } else {
        format!("consider removing `{param_name}` or referring to it in a field")
    };
    err.help(msg);

    if matches!(param.kind, hir::GenericParamKind::Type { .. }) && !has_explicit_bounds {
        err.help(format!(
            "if you intended `{0}` to be a const parameter, use `const {0}: usize` instead",
            param_name
        ));
    }
    err.emit()
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
            if let ty::PredicateKind::Clause(ty::ClauseKind::WellFormed(..)) =
                pred.kind().skip_binder()
            {
                continue;
            }
            // Match the existing behavior.
            if pred.is_global() && !pred.has_late_bound_vars() {
                let pred = self.normalize(span, None, pred);
                let hir_node = tcx.hir().find_by_def_id(self.body_def_id);

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

fn check_mod_type_wf(tcx: TyCtxt<'_>, module: LocalDefId) {
    let items = tcx.hir_module_items(module);
    items.par_items(|item| tcx.ensure().check_well_formed(item.owner_id));
    items.par_impl_items(|item| tcx.ensure().check_well_formed(item.owner_id));
    items.par_trait_items(|item| tcx.ensure().check_well_formed(item.owner_id));
    items.par_foreign_items(|item| tcx.ensure().check_well_formed(item.owner_id));
}

fn error_392(
    tcx: TyCtxt<'_>,
    span: Span,
    param_name: Symbol,
) -> DiagnosticBuilder<'_, ErrorGuaranteed> {
    let mut err = struct_span_err!(tcx.sess, span, E0392, "parameter `{param_name}` is never used");
    err.span_label(span, "unused parameter");
    err
}

pub fn provide(providers: &mut Providers) {
    *providers = Providers { check_mod_type_wf, check_well_formed, ..*providers };
}
