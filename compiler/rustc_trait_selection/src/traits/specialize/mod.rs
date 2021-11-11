//! Logic and data structures related to impl specialization, explained in
//! greater detail below.
//!
//! At the moment, this implementation support only the simple "chain" rule:
//! If any two impls overlap, one must be a strict subset of the other.
//!
//! See the [rustc dev guide] for a bit more detail on how specialization
//! fits together with the rest of the trait machinery.
//!
//! [rustc dev guide]: https://rustc-dev-guide.rust-lang.org/traits/specialization.html

pub mod specialization_graph;
use specialization_graph::GraphExt;

use crate::infer::{InferCtxt, InferOk, TyCtxtInferExt};
use crate::traits::select::IntercrateAmbiguityCause;
use crate::traits::{self, coherence, FutureCompatOverlapErrorKind, ObligationCause, TraitEngine};
use rustc_data_structures::fx::FxHashSet;
use rustc_errors::struct_span_err;
use rustc_hir::def_id::{DefId, LocalDefId};
use rustc_middle::lint::LintDiagnosticBuilder;
use rustc_middle::ty::subst::{InternalSubsts, Subst, SubstsRef};
use rustc_middle::ty::{self, TyCtxt};
use rustc_session::lint::builtin::COHERENCE_LEAK_CHECK;
use rustc_session::lint::builtin::ORDER_DEPENDENT_TRAIT_OBJECTS;
use rustc_span::DUMMY_SP;

use super::util::impl_trait_ref_and_oblig;
use super::{FulfillmentContext, SelectionContext};

/// Information pertinent to an overlapping impl error.
#[derive(Debug)]
pub struct OverlapError {
    pub with_impl: DefId,
    pub trait_desc: String,
    pub self_desc: Option<String>,
    pub intercrate_ambiguity_causes: Vec<IntercrateAmbiguityCause>,
    pub involves_placeholder: bool,
}

/// Given a subst for the requested impl, translate it to a subst
/// appropriate for the actual item definition (whether it be in that impl,
/// a parent impl, or the trait).
///
/// When we have selected one impl, but are actually using item definitions from
/// a parent impl providing a default, we need a way to translate between the
/// type parameters of the two impls. Here the `source_impl` is the one we've
/// selected, and `source_substs` is a substitution of its generics.
/// And `target_node` is the impl/trait we're actually going to get the
/// definition from. The resulting substitution will map from `target_node`'s
/// generics to `source_impl`'s generics as instantiated by `source_subst`.
///
/// For example, consider the following scenario:
///
/// ```rust
/// trait Foo { ... }
/// impl<T, U> Foo for (T, U) { ... }  // target impl
/// impl<V> Foo for (V, V) { ... }     // source impl
/// ```
///
/// Suppose we have selected "source impl" with `V` instantiated with `u32`.
/// This function will produce a substitution with `T` and `U` both mapping to `u32`.
///
/// where-clauses add some trickiness here, because they can be used to "define"
/// an argument indirectly:
///
/// ```rust
/// impl<'a, I, T: 'a> Iterator for Cloned<I>
///    where I: Iterator<Item = &'a T>, T: Clone
/// ```
///
/// In a case like this, the substitution for `T` is determined indirectly,
/// through associated type projection. We deal with such cases by using
/// *fulfillment* to relate the two impls, requiring that all projections are
/// resolved.
pub fn translate_substs<'a, 'tcx>(
    infcx: &InferCtxt<'a, 'tcx>,
    param_env: ty::ParamEnv<'tcx>,
    source_impl: DefId,
    source_substs: SubstsRef<'tcx>,
    target_node: specialization_graph::Node,
) -> SubstsRef<'tcx> {
    debug!(
        "translate_substs({:?}, {:?}, {:?}, {:?})",
        param_env, source_impl, source_substs, target_node
    );
    let source_trait_ref =
        infcx.tcx.impl_trait_ref(source_impl).unwrap().subst(infcx.tcx, &source_substs);

    // translate the Self and Param parts of the substitution, since those
    // vary across impls
    let target_substs = match target_node {
        specialization_graph::Node::Impl(target_impl) => {
            // no need to translate if we're targeting the impl we started with
            if source_impl == target_impl {
                return source_substs;
            }

            fulfill_implication(infcx, param_env, source_trait_ref, target_impl).unwrap_or_else(
                |_| {
                    bug!(
                        "When translating substitutions for specialization, the expected \
                         specialization failed to hold"
                    )
                },
            )
        }
        specialization_graph::Node::Trait(..) => source_trait_ref.substs,
    };

    // directly inherent the method generics, since those do not vary across impls
    source_substs.rebase_onto(infcx.tcx, source_impl, target_substs)
}

/// Is `impl1` a specialization of `impl2`?
///
/// Specialization is determined by the sets of types to which the impls apply;
/// `impl1` specializes `impl2` if it applies to a subset of the types `impl2` applies
/// to.
pub(super) fn specializes(tcx: TyCtxt<'_>, (impl1_def_id, impl2_def_id): (DefId, DefId)) -> bool {
    debug!("specializes({:?}, {:?})", impl1_def_id, impl2_def_id);

    // The feature gate should prevent introducing new specializations, but not
    // taking advantage of upstream ones.
    let features = tcx.features();
    let specialization_enabled = features.specialization || features.min_specialization;
    if !specialization_enabled && (impl1_def_id.is_local() || impl2_def_id.is_local()) {
        return false;
    }

    // We determine whether there's a subset relationship by:
    //
    // - replacing bound vars with placeholders in impl1,
    // - assuming the where clauses for impl1,
    // - instantiating impl2 with fresh inference variables,
    // - unifying,
    // - attempting to prove the where clauses for impl2
    //
    // The last three steps are encapsulated in `fulfill_implication`.
    //
    // See RFC 1210 for more details and justification.

    // Currently we do not allow e.g., a negative impl to specialize a positive one
    if tcx.impl_polarity(impl1_def_id) != tcx.impl_polarity(impl2_def_id) {
        return false;
    }

    // create a parameter environment corresponding to a (placeholder) instantiation of impl1
    let penv = tcx.param_env(impl1_def_id);
    let impl1_trait_ref = tcx.impl_trait_ref(impl1_def_id).unwrap();

    // Create an infcx, taking the predicates of impl1 as assumptions:
    tcx.infer_ctxt().enter(|infcx| {
        // Normalize the trait reference. The WF rules ought to ensure
        // that this always succeeds.
        let impl1_trait_ref = match traits::fully_normalize(
            &infcx,
            FulfillmentContext::new(),
            ObligationCause::dummy(),
            penv,
            impl1_trait_ref,
        ) {
            Ok(impl1_trait_ref) => impl1_trait_ref,
            Err(err) => {
                bug!("failed to fully normalize {:?}: {:?}", impl1_trait_ref, err);
            }
        };

        // Attempt to prove that impl2 applies, given all of the above.
        fulfill_implication(&infcx, penv, impl1_trait_ref, impl2_def_id).is_ok()
    })
}

/// Attempt to fulfill all obligations of `target_impl` after unification with
/// `source_trait_ref`. If successful, returns a substitution for *all* the
/// generics of `target_impl`, including both those needed to unify with
/// `source_trait_ref` and those whose identity is determined via a where
/// clause in the impl.
fn fulfill_implication<'a, 'tcx>(
    infcx: &InferCtxt<'a, 'tcx>,
    param_env: ty::ParamEnv<'tcx>,
    source_trait_ref: ty::TraitRef<'tcx>,
    target_impl: DefId,
) -> Result<SubstsRef<'tcx>, ()> {
    debug!(
        "fulfill_implication({:?}, trait_ref={:?} |- {:?} applies)",
        param_env, source_trait_ref, target_impl
    );

    let selcx = &mut SelectionContext::new(&infcx);
    let target_substs = infcx.fresh_substs_for_item(DUMMY_SP, target_impl);
    let (target_trait_ref, obligations) =
        impl_trait_ref_and_oblig(selcx, param_env, target_impl, target_substs);

    // do the impls unify? If not, no specialization.
    let more_obligations =
        match infcx.at(&ObligationCause::dummy(), param_env).eq(source_trait_ref, target_trait_ref)
        {
            Ok(InferOk { obligations, .. }) => obligations,
            Err(_) => {
                debug!(
                    "fulfill_implication: {:?} does not unify with {:?}",
                    source_trait_ref, target_trait_ref
                );
                return Err(());
            }
        };

    // attempt to prove all of the predicates for impl2 given those for impl1
    // (which are packed up in penv)

    infcx.save_and_restore_in_snapshot_flag(|infcx| {
        // If we came from `translate_substs`, we already know that the
        // predicates for our impl hold (after all, we know that a more
        // specialized impl holds, so our impl must hold too), and
        // we only want to process the projections to determine the
        // the types in our substs using RFC 447, so we can safely
        // ignore region obligations, which allows us to avoid threading
        // a node-id to assign them with.
        //
        // If we came from specialization graph construction, then
        // we already make a mockery out of the region system, so
        // why not ignore them a bit earlier?
        let mut fulfill_cx = FulfillmentContext::new_ignoring_regions();
        for oblig in obligations.chain(more_obligations) {
            fulfill_cx.register_predicate_obligation(&infcx, oblig);
        }
        match fulfill_cx.select_all_or_error(infcx).as_slice() {
            [] => {
                debug!(
                    "fulfill_implication: an impl for {:?} specializes {:?}",
                    source_trait_ref, target_trait_ref
                );

                // Now resolve the *substitution* we built for the target earlier, replacing
                // the inference variables inside with whatever we got from fulfillment.
                Ok(infcx.resolve_vars_if_possible(target_substs))
            }
            errors => {
                // no dice!
                debug!(
                    "fulfill_implication: for impls on {:?} and {:?}, \
                     could not fulfill: {:?} given {:?}",
                    source_trait_ref,
                    target_trait_ref,
                    errors,
                    param_env.caller_bounds()
                );
                Err(())
            }
        }
    })
}

// Query provider for `specialization_graph_of`.
pub(super) fn specialization_graph_provider(
    tcx: TyCtxt<'_>,
    trait_id: DefId,
) -> specialization_graph::Graph {
    let mut sg = specialization_graph::Graph::new();

    let mut trait_impls: Vec<_> = tcx.all_impls(trait_id).collect();

    // The coherence checking implementation seems to rely on impls being
    // iterated over (roughly) in definition order, so we are sorting by
    // negated `CrateNum` (so remote definitions are visited first) and then
    // by a flattened version of the `DefIndex`.
    trait_impls
        .sort_unstable_by_key(|def_id| (-(def_id.krate.as_u32() as i64), def_id.index.index()));

    for impl_def_id in trait_impls {
        if let Some(impl_def_id) = impl_def_id.as_local() {
            // This is where impl overlap checking happens:
            let insert_result = sg.insert(tcx, impl_def_id.to_def_id());
            // Report error if there was one.
            let (overlap, used_to_be_allowed) = match insert_result {
                Err(overlap) => (Some(overlap), None),
                Ok(Some(overlap)) => (Some(overlap.error), Some(overlap.kind)),
                Ok(None) => (None, None),
            };

            if let Some(overlap) = overlap {
                report_overlap_conflict(tcx, overlap, impl_def_id, used_to_be_allowed, &mut sg);
            }
        } else {
            let parent = tcx.impl_parent(impl_def_id).unwrap_or(trait_id);
            sg.record_impl_from_cstore(tcx, parent, impl_def_id)
        }
    }

    sg
}

// This function is only used when
// encountering errors and inlining
// it negatively impacts perf.
#[cold]
#[inline(never)]
fn report_overlap_conflict(
    tcx: TyCtxt<'_>,
    overlap: OverlapError,
    impl_def_id: LocalDefId,
    used_to_be_allowed: Option<FutureCompatOverlapErrorKind>,
    sg: &mut specialization_graph::Graph,
) {
    let impl_polarity = tcx.impl_polarity(impl_def_id.to_def_id());
    let other_polarity = tcx.impl_polarity(overlap.with_impl);
    match (impl_polarity, other_polarity) {
        (ty::ImplPolarity::Negative, ty::ImplPolarity::Positive) => {
            report_negative_positive_conflict(
                tcx,
                &overlap,
                impl_def_id,
                impl_def_id.to_def_id(),
                overlap.with_impl,
                sg,
            );
        }

        (ty::ImplPolarity::Positive, ty::ImplPolarity::Negative) => {
            report_negative_positive_conflict(
                tcx,
                &overlap,
                impl_def_id,
                overlap.with_impl,
                impl_def_id.to_def_id(),
                sg,
            );
        }

        _ => {
            report_conflicting_impls(tcx, overlap, impl_def_id, used_to_be_allowed, sg);
        }
    }
}

fn report_negative_positive_conflict(
    tcx: TyCtxt<'_>,
    overlap: &OverlapError,
    local_impl_def_id: LocalDefId,
    negative_impl_def_id: DefId,
    positive_impl_def_id: DefId,
    sg: &mut specialization_graph::Graph,
) {
    let impl_span = tcx
        .sess
        .source_map()
        .guess_head_span(tcx.span_of_impl(local_impl_def_id.to_def_id()).unwrap());

    let mut err = struct_span_err!(
        tcx.sess,
        impl_span,
        E0751,
        "found both positive and negative implementation of trait `{}`{}:",
        overlap.trait_desc,
        overlap.self_desc.clone().map_or_else(String::new, |ty| format!(" for type `{}`", ty))
    );

    match tcx.span_of_impl(negative_impl_def_id) {
        Ok(span) => {
            err.span_label(
                tcx.sess.source_map().guess_head_span(span),
                "negative implementation here".to_string(),
            );
        }
        Err(cname) => {
            err.note(&format!("negative implementation in crate `{}`", cname));
        }
    }

    match tcx.span_of_impl(positive_impl_def_id) {
        Ok(span) => {
            err.span_label(
                tcx.sess.source_map().guess_head_span(span),
                "positive implementation here".to_string(),
            );
        }
        Err(cname) => {
            err.note(&format!("positive implementation in crate `{}`", cname));
        }
    }

    sg.has_errored = true;
    err.emit();
}

fn report_conflicting_impls(
    tcx: TyCtxt<'_>,
    overlap: OverlapError,
    impl_def_id: LocalDefId,
    used_to_be_allowed: Option<FutureCompatOverlapErrorKind>,
    sg: &mut specialization_graph::Graph,
) {
    let impl_span =
        tcx.sess.source_map().guess_head_span(tcx.span_of_impl(impl_def_id.to_def_id()).unwrap());

    // Work to be done after we've built the DiagnosticBuilder. We have to define it
    // now because the struct_lint methods don't return back the DiagnosticBuilder
    // that's passed in.
    let decorate = |err: LintDiagnosticBuilder<'_>| {
        let msg = format!(
            "conflicting implementations of trait `{}`{}{}",
            overlap.trait_desc,
            overlap
                .self_desc
                .clone()
                .map_or_else(String::new, |ty| { format!(" for type `{}`", ty) }),
            match used_to_be_allowed {
                Some(FutureCompatOverlapErrorKind::Issue33140) => ": (E0119)",
                _ => "",
            }
        );
        let mut err = err.build(&msg);
        match tcx.span_of_impl(overlap.with_impl) {
            Ok(span) => {
                err.span_label(
                    tcx.sess.source_map().guess_head_span(span),
                    "first implementation here".to_string(),
                );

                err.span_label(
                    impl_span,
                    format!(
                        "conflicting implementation{}",
                        overlap.self_desc.map_or_else(String::new, |ty| format!(" for `{}`", ty))
                    ),
                );
            }
            Err(cname) => {
                let msg = match to_pretty_impl_header(tcx, overlap.with_impl) {
                    Some(s) => format!("conflicting implementation in crate `{}`:\n- {}", cname, s),
                    None => format!("conflicting implementation in crate `{}`", cname),
                };
                err.note(&msg);
            }
        }

        for cause in &overlap.intercrate_ambiguity_causes {
            cause.add_intercrate_ambiguity_hint(&mut err);
        }

        if overlap.involves_placeholder {
            coherence::add_placeholder_note(&mut err);
        }
        err.emit()
    };

    match used_to_be_allowed {
        None => {
            sg.has_errored = true;
            if overlap.with_impl.is_local() || !tcx.orphan_check_crate(()).contains(&impl_def_id) {
                let err = struct_span_err!(tcx.sess, impl_span, E0119, "");
                decorate(LintDiagnosticBuilder::new(err));
            } else {
                tcx.sess.delay_span_bug(impl_span, "impl should have failed the orphan check");
            }
        }
        Some(kind) => {
            let lint = match kind {
                FutureCompatOverlapErrorKind::Issue33140 => ORDER_DEPENDENT_TRAIT_OBJECTS,
                FutureCompatOverlapErrorKind::LeakCheck => COHERENCE_LEAK_CHECK,
            };
            tcx.struct_span_lint_hir(
                lint,
                tcx.hir().local_def_id_to_hir_id(impl_def_id),
                impl_span,
                decorate,
            )
        }
    };
}

/// Recovers the "impl X for Y" signature from `impl_def_id` and returns it as a
/// string.
crate fn to_pretty_impl_header(tcx: TyCtxt<'_>, impl_def_id: DefId) -> Option<String> {
    use std::fmt::Write;

    let trait_ref = tcx.impl_trait_ref(impl_def_id)?;
    let mut w = "impl".to_owned();

    let substs = InternalSubsts::identity_for_item(tcx, impl_def_id);

    // FIXME: Currently only handles ?Sized.
    //        Needs to support ?Move and ?DynSized when they are implemented.
    let mut types_without_default_bounds = FxHashSet::default();
    let sized_trait = tcx.lang_items().sized_trait();

    if !substs.is_noop() {
        types_without_default_bounds.extend(substs.types());
        w.push('<');
        w.push_str(
            &substs
                .iter()
                .map(|k| k.to_string())
                .filter(|k| k != "'_")
                .collect::<Vec<_>>()
                .join(", "),
        );
        w.push('>');
    }

    write!(w, " {} for {}", trait_ref.print_only_trait_path(), tcx.type_of(impl_def_id)).unwrap();

    // The predicates will contain default bounds like `T: Sized`. We need to
    // remove these bounds, and add `T: ?Sized` to any untouched type parameters.
    let predicates = tcx.predicates_of(impl_def_id).predicates;
    let mut pretty_predicates =
        Vec::with_capacity(predicates.len() + types_without_default_bounds.len());

    for (p, _) in predicates {
        if let Some(poly_trait_ref) = p.to_opt_poly_trait_ref() {
            if Some(poly_trait_ref.value.def_id()) == sized_trait {
                types_without_default_bounds.remove(poly_trait_ref.value.self_ty().skip_binder());
                continue;
            }
        }
        pretty_predicates.push(p.to_string());
    }

    pretty_predicates
        .extend(types_without_default_bounds.iter().map(|ty| format!("{}: ?Sized", ty)));

    if !pretty_predicates.is_empty() {
        write!(w, "\n  where {}", pretty_predicates.join(", ")).unwrap();
    }

    w.push(';');
    Some(w)
}
