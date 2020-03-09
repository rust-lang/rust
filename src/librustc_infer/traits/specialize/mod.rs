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
use rustc::lint::LintDiagnosticBuilder;
use rustc::ty::subst::{InternalSubsts, Subst, SubstsRef};
use rustc::ty::{self, TyCtxt, TypeFoldable};
use rustc_data_structures::fx::FxHashSet;
use rustc_errors::struct_span_err;
use rustc_hir::def_id::DefId;
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

/// Given a selected impl described by `impl_data`, returns the
/// definition and substitutions for the method with the name `name`
/// the kind `kind`, and trait method substitutions `substs`, in
/// that impl, a less specialized impl, or the trait default,
/// whichever applies.
pub fn find_associated_item<'tcx>(
    tcx: TyCtxt<'tcx>,
    param_env: ty::ParamEnv<'tcx>,
    item: &ty::AssocItem,
    substs: SubstsRef<'tcx>,
    impl_data: &super::VtableImplData<'tcx, ()>,
) -> (DefId, SubstsRef<'tcx>) {
    debug!("find_associated_item({:?}, {:?}, {:?}, {:?})", param_env, item, substs, impl_data);
    assert!(!substs.needs_infer());

    let trait_def_id = tcx.trait_id_of_impl(impl_data.impl_def_id).unwrap();
    let trait_def = tcx.trait_def(trait_def_id);

    let ancestors = trait_def.ancestors(tcx, impl_data.impl_def_id);
    match ancestors.leaf_def(tcx, item.ident, item.kind) {
        Some(node_item) => {
            let substs = tcx.infer_ctxt().enter(|infcx| {
                let param_env = param_env.with_reveal_all();
                let substs = substs.rebase_onto(tcx, trait_def_id, impl_data.substs);
                let substs = translate_substs(
                    &infcx,
                    param_env,
                    impl_data.impl_def_id,
                    substs,
                    node_item.node,
                );
                infcx.tcx.erase_regions(&substs)
            });
            (node_item.item.def_id, substs)
        }
        None => bug!("{:?} not found in {:?}", item, impl_data.impl_def_id),
    }
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
    if !tcx.features().specialization && (impl1_def_id.is_local() || impl2_def_id.is_local()) {
        return false;
    }

    // We determine whether there's a subset relationship by:
    //
    // - skolemizing impl1,
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

    // Create a infcx, taking the predicates of impl1 as assumptions:
    tcx.infer_ctxt().enter(|infcx| {
        // Normalize the trait reference. The WF rules ought to ensure
        // that this always succeeds.
        let impl1_trait_ref = match traits::fully_normalize(
            &infcx,
            FulfillmentContext::new(),
            ObligationCause::dummy(),
            penv,
            &impl1_trait_ref,
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
    let (target_trait_ref, mut obligations) =
        impl_trait_ref_and_oblig(selcx, param_env, target_impl, target_substs);
    debug!(
        "fulfill_implication: target_trait_ref={:?}, obligations={:?}",
        target_trait_ref, obligations
    );

    // do the impls unify? If not, no specialization.
    match infcx.at(&ObligationCause::dummy(), param_env).eq(source_trait_ref, target_trait_ref) {
        Ok(InferOk { obligations: o, .. }) => {
            obligations.extend(o);
        }
        Err(_) => {
            debug!(
                "fulfill_implication: {:?} does not unify with {:?}",
                source_trait_ref, target_trait_ref
            );
            return Err(());
        }
    }

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
        for oblig in obligations.into_iter() {
            fulfill_cx.register_predicate_obligation(&infcx, oblig);
        }
        match fulfill_cx.select_all_or_error(infcx) {
            Err(errors) => {
                // no dice!
                debug!(
                    "fulfill_implication: for impls on {:?} and {:?}, \
                     could not fulfill: {:?} given {:?}",
                    source_trait_ref, target_trait_ref, errors, param_env.caller_bounds
                );
                Err(())
            }

            Ok(()) => {
                debug!(
                    "fulfill_implication: an impl for {:?} specializes {:?}",
                    source_trait_ref, target_trait_ref
                );

                // Now resolve the *substitution* we built for the target earlier, replacing
                // the inference variables inside with whatever we got from fulfillment.
                Ok(infcx.resolve_vars_if_possible(&target_substs))
            }
        }
    })
}

// Query provider for `specialization_graph_of`.
pub(super) fn specialization_graph_provider(
    tcx: TyCtxt<'_>,
    trait_id: DefId,
) -> &specialization_graph::Graph {
    let mut sg = specialization_graph::Graph::new();

    let mut trait_impls = tcx.all_impls(trait_id);

    // The coherence checking implementation seems to rely on impls being
    // iterated over (roughly) in definition order, so we are sorting by
    // negated `CrateNum` (so remote definitions are visited first) and then
    // by a flattened version of the `DefIndex`.
    trait_impls
        .sort_unstable_by_key(|def_id| (-(def_id.krate.as_u32() as i64), def_id.index.index()));

    for impl_def_id in trait_impls {
        if impl_def_id.is_local() {
            // This is where impl overlap checking happens:
            let insert_result = sg.insert(tcx, impl_def_id);
            // Report error if there was one.
            let (overlap, used_to_be_allowed) = match insert_result {
                Err(overlap) => (Some(overlap), None),
                Ok(Some(overlap)) => (Some(overlap.error), Some(overlap.kind)),
                Ok(None) => (None, None),
            };

            if let Some(overlap) = overlap {
                let impl_span =
                    tcx.sess.source_map().def_span(tcx.span_of_impl(impl_def_id).unwrap());

                // Work to be done after we've built the DiagnosticBuilder. We have to define it
                // now because the struct_lint methods don't return back the DiagnosticBuilder
                // that's passed in.
                let decorate = |err: LintDiagnosticBuilder<'_>| {
                    let msg = format!(
                        "conflicting implementations of trait `{}`{}:{}",
                        overlap.trait_desc,
                        overlap
                            .self_desc
                            .clone()
                            .map_or(String::new(), |ty| { format!(" for type `{}`", ty) }),
                        match used_to_be_allowed {
                            Some(FutureCompatOverlapErrorKind::Issue33140) => " (E0119)",
                            _ => "",
                        }
                    );
                    let mut err = err.build(&msg);
                    match tcx.span_of_impl(overlap.with_impl) {
                        Ok(span) => {
                            err.span_label(
                                tcx.sess.source_map().def_span(span),
                                "first implementation here".to_string(),
                            );

                            err.span_label(
                                impl_span,
                                format!(
                                    "conflicting implementation{}",
                                    overlap
                                        .self_desc
                                        .map_or(String::new(), |ty| format!(" for `{}`", ty))
                                ),
                            );
                        }
                        Err(cname) => {
                            let msg = match to_pretty_impl_header(tcx, overlap.with_impl) {
                                Some(s) => format!(
                                    "conflicting implementation in crate `{}`:\n- {}",
                                    cname, s
                                ),
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
                        let err = struct_span_err!(tcx.sess, impl_span, E0119, "");
                        decorate(LintDiagnosticBuilder::new(err));
                    }
                    Some(kind) => {
                        let lint = match kind {
                            FutureCompatOverlapErrorKind::Issue33140 => {
                                ORDER_DEPENDENT_TRAIT_OBJECTS
                            }
                            FutureCompatOverlapErrorKind::LeakCheck => COHERENCE_LEAK_CHECK,
                        };
                        tcx.struct_span_lint_hir(
                            lint,
                            tcx.hir().as_local_hir_id(impl_def_id).unwrap(),
                            impl_span,
                            decorate,
                        )
                    }
                };
            }
        } else {
            let parent = tcx.impl_parent(impl_def_id).unwrap_or(trait_id);
            sg.record_impl_from_cstore(tcx, parent, impl_def_id)
        }
    }

    tcx.arena.alloc(sg)
}

/// Recovers the "impl X for Y" signature from `impl_def_id` and returns it as a
/// string.
fn to_pretty_impl_header(tcx: TyCtxt<'_>, impl_def_id: DefId) -> Option<String> {
    use std::fmt::Write;

    let trait_ref = if let Some(tr) = tcx.impl_trait_ref(impl_def_id) {
        tr
    } else {
        return None;
    };

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
            if Some(poly_trait_ref.def_id()) == sized_trait {
                types_without_default_bounds.remove(poly_trait_ref.self_ty());
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
