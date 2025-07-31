use std::cell::OnceCell;
use std::ops::ControlFlow;

use rustc_data_structures::fx::FxHashSet;
use rustc_data_structures::graph::iterate::DepthFirstSearch;
use rustc_data_structures::graph::vec_graph::VecGraph;
use rustc_data_structures::graph::{self};
use rustc_data_structures::unord::{UnordBag, UnordMap, UnordSet};
use rustc_hir as hir;
use rustc_hir::HirId;
use rustc_hir::def::{DefKind, Res};
use rustc_hir::def_id::DefId;
use rustc_hir::intravisit::{InferKind, Visitor};
use rustc_middle::ty::{self, Ty, TyCtxt, TypeSuperVisitable, TypeVisitable};
use rustc_session::lint;
use rustc_span::def_id::LocalDefId;
use rustc_span::{DUMMY_SP, Span};
use rustc_trait_selection::traits::{ObligationCause, ObligationCtxt};
use tracing::debug;

use crate::typeck_root_ctxt::InferVarInfo;
use crate::{FnCtxt, errors};

#[derive(Copy, Clone)]
pub(crate) enum DivergingFallbackBehavior {
    /// Always fallback to `()` (aka "always spontaneous decay")
    ToUnit,
    /// Sometimes fallback to `!`, but mainly fallback to `()` so that most of the crates are not broken.
    ContextDependent,
    /// Always fallback to `!` (which should be equivalent to never falling back + not making
    /// never-to-any coercions unless necessary)
    ToNever,
    /// Don't fallback at all
    NoFallback,
}

impl<'tcx> FnCtxt<'_, 'tcx> {
    /// Performs type inference fallback, setting `FnCtxt::fallback_has_occurred`
    /// if fallback has occurred.
    pub(super) fn type_inference_fallback(&self) {
        debug!(
            "type-inference-fallback start obligations: {:#?}",
            self.fulfillment_cx.borrow_mut().pending_obligations()
        );

        // All type checking constraints were added, try to fallback unsolved variables.
        self.select_obligations_where_possible(|_| {});

        debug!(
            "type-inference-fallback post selection obligations: {:#?}",
            self.fulfillment_cx.borrow_mut().pending_obligations()
        );

        let fallback_occurred = self.fallback_types();

        if !fallback_occurred {
            return;
        }

        // We now see if we can make progress. This might cause us to
        // unify inference variables for opaque types, since we may
        // have unified some other type variables during the first
        // phase of fallback. This means that we only replace
        // inference variables with their underlying opaque types as a
        // last resort.
        //
        // In code like this:
        //
        // ```rust
        // type MyType = impl Copy;
        // fn produce() -> MyType { true }
        // fn bad_produce() -> MyType { panic!() }
        // ```
        //
        // we want to unify the opaque inference variable in `bad_produce`
        // with the diverging fallback for `panic!` (e.g. `()` or `!`).
        // This will produce a nice error message about conflicting concrete
        // types for `MyType`.
        //
        // If we had tried to fallback the opaque inference variable to `MyType`,
        // we will generate a confusing type-check error that does not explicitly
        // refer to opaque types.
        self.select_obligations_where_possible(|_| {});
    }

    fn fallback_types(&self) -> bool {
        // Check if we have any unresolved variables. If not, no need for fallback.
        let unresolved_variables = self.unresolved_variables();

        if unresolved_variables.is_empty() {
            return false;
        }

        let diverging_fallback = self
            .calculate_diverging_fallback(&unresolved_variables, self.diverging_fallback_behavior);

        // We do fallback in two passes, to try to generate
        // better error messages.
        // The first time, we do *not* replace opaque types.
        let mut fallback_occurred = false;
        for ty in unresolved_variables {
            debug!("unsolved_variable = {:?}", ty);
            fallback_occurred |= self.fallback_if_possible(ty, &diverging_fallback);
        }

        fallback_occurred
    }

    // Tries to apply a fallback to `ty` if it is an unsolved variable.
    //
    // - Unconstrained ints are replaced with `i32`.
    //
    // - Unconstrained floats are replaced with `f64`.
    //
    // - Non-numerics may get replaced with `()` or `!`, depending on
    //   how they were categorized by `calculate_diverging_fallback`
    //   (and the setting of `#![feature(never_type_fallback)]`).
    //
    // Fallback becomes very dubious if we have encountered
    // type-checking errors. In that case, fallback to Error.
    //
    // Sets `FnCtxt::fallback_has_occurred` if fallback is performed
    // during this call.
    fn fallback_if_possible(
        &self,
        ty: Ty<'tcx>,
        diverging_fallback: &UnordMap<Ty<'tcx>, Ty<'tcx>>,
    ) -> bool {
        // Careful: we do NOT shallow-resolve `ty`. We know that `ty`
        // is an unsolved variable, and we determine its fallback
        // based solely on how it was created, not what other type
        // variables it may have been unified with since then.
        //
        // The reason this matters is that other attempts at fallback
        // may (in principle) conflict with this fallback, and we wish
        // to generate a type error in that case. (However, this
        // actually isn't true right now, because we're only using the
        // builtin fallback rules. This would be true if we were using
        // user-supplied fallbacks. But it's still useful to write the
        // code to detect bugs.)
        //
        // (Note though that if we have a general type variable `?T`
        // that is then unified with an integer type variable `?I`
        // that ultimately never gets resolved to a special integral
        // type, `?T` is not considered unsolved, but `?I` is. The
        // same is true for float variables.)
        let fallback = match ty.kind() {
            _ if let Some(e) = self.tainted_by_errors() => Ty::new_error(self.tcx, e),
            ty::Infer(ty::IntVar(_)) => self.tcx.types.i32,
            ty::Infer(ty::FloatVar(_)) => self.tcx.types.f64,
            _ => match diverging_fallback.get(&ty) {
                Some(&fallback_ty) => fallback_ty,
                None => return false,
            },
        };
        debug!("fallback_if_possible(ty={:?}): defaulting to `{:?}`", ty, fallback);

        let span = ty.ty_vid().map_or(DUMMY_SP, |vid| self.infcx.type_var_origin(vid).span);
        self.demand_eqtype(span, ty, fallback);
        self.fallback_has_occurred.set(true);
        true
    }

    /// The "diverging fallback" system is rather complicated. This is
    /// a result of our need to balance 'do the right thing' with
    /// backwards compatibility.
    ///
    /// "Diverging" type variables are variables created when we
    /// coerce a `!` type into an unbound type variable `?X`. If they
    /// never wind up being constrained, the "right and natural" thing
    /// is that `?X` should "fallback" to `!`. This means that e.g. an
    /// expression like `Some(return)` will ultimately wind up with a
    /// type like `Option<!>` (presuming it is not assigned or
    /// constrained to have some other type).
    ///
    /// However, the fallback used to be `()` (before the `!` type was
    /// added). Moreover, there are cases where the `!` type 'leaks
    /// out' from dead code into type variables that affect live
    /// code. The most common case is something like this:
    ///
    /// ```rust
    /// # fn foo() -> i32 { 4 }
    /// match foo() {
    ///     22 => Default::default(), // call this type `?D`
    ///     _ => return, // return has type `!`
    /// } // call the type of this match `?M`
    /// ```
    ///
    /// Here, coercing the type `!` into `?M` will create a diverging
    /// type variable `?X` where `?X <: ?M`. We also have that `?D <:
    /// ?M`. If `?M` winds up unconstrained, then `?X` will
    /// fallback. If it falls back to `!`, then all the type variables
    /// will wind up equal to `!` -- this includes the type `?D`
    /// (since `!` doesn't implement `Default`, we wind up a "trait
    /// not implemented" error in code like this). But since the
    /// original fallback was `()`, this code used to compile with `?D
    /// = ()`. This is somewhat surprising, since `Default::default()`
    /// on its own would give an error because the types are
    /// insufficiently constrained.
    ///
    /// Our solution to this dilemma is to modify diverging variables
    /// so that they can *either* fallback to `!` (the default) or to
    /// `()` (the backwards compatibility case). We decide which
    /// fallback to use based on whether there is a coercion pattern
    /// like this:
    ///
    /// ```ignore (not-rust)
    /// ?Diverging -> ?V
    /// ?NonDiverging -> ?V
    /// ?V != ?NonDiverging
    /// ```
    ///
    /// Here `?Diverging` represents some diverging type variable and
    /// `?NonDiverging` represents some non-diverging type
    /// variable. `?V` can be any type variable (diverging or not), so
    /// long as it is not equal to `?NonDiverging`.
    ///
    /// Intuitively, what we are looking for is a case where a
    /// "non-diverging" type variable (like `?M` in our example above)
    /// is coerced *into* some variable `?V` that would otherwise
    /// fallback to `!`. In that case, we make `?V` fallback to `!`,
    /// along with anything that would flow into `?V`.
    ///
    /// The algorithm we use:
    /// * Identify all variables that are coerced *into* by a
    ///   diverging variable. Do this by iterating over each
    ///   diverging, unsolved variable and finding all variables
    ///   reachable from there. Call that set `D`.
    /// * Walk over all unsolved, non-diverging variables, and find
    ///   any variable that has an edge into `D`.
    fn calculate_diverging_fallback(
        &self,
        unresolved_variables: &[Ty<'tcx>],
        behavior: DivergingFallbackBehavior,
    ) -> UnordMap<Ty<'tcx>, Ty<'tcx>> {
        debug!("calculate_diverging_fallback({:?})", unresolved_variables);

        // Construct a coercion graph where an edge `A -> B` indicates
        // a type variable is that is coerced
        let coercion_graph = self.create_coercion_graph();

        // Extract the unsolved type inference variable vids; note that some
        // unsolved variables are integer/float variables and are excluded.
        let unsolved_vids = unresolved_variables.iter().filter_map(|ty| ty.ty_vid());

        // Compute the diverging root vids D -- that is, the root vid of
        // those type variables that (a) are the target of a coercion from
        // a `!` type and (b) have not yet been solved.
        //
        // These variables are the ones that are targets for fallback to
        // either `!` or `()`.
        let diverging_roots: UnordSet<ty::TyVid> = self
            .diverging_type_vars
            .borrow()
            .items()
            .map(|&ty| self.shallow_resolve(ty))
            .filter_map(|ty| ty.ty_vid())
            .map(|vid| self.root_var(vid))
            .collect();
        debug!(
            "calculate_diverging_fallback: diverging_type_vars={:?}",
            self.diverging_type_vars.borrow()
        );
        debug!("calculate_diverging_fallback: diverging_roots={:?}", diverging_roots);

        // Find all type variables that are reachable from a diverging
        // type variable. These will typically default to `!`, unless
        // we find later that they are *also* reachable from some
        // other type variable outside this set.
        let mut roots_reachable_from_diverging = DepthFirstSearch::new(&coercion_graph);
        let mut diverging_vids = vec![];
        let mut non_diverging_vids = vec![];
        for unsolved_vid in unsolved_vids {
            let root_vid = self.root_var(unsolved_vid);
            debug!(
                "calculate_diverging_fallback: unsolved_vid={:?} root_vid={:?} diverges={:?}",
                unsolved_vid,
                root_vid,
                diverging_roots.contains(&root_vid),
            );
            if diverging_roots.contains(&root_vid) {
                diverging_vids.push(unsolved_vid);
                roots_reachable_from_diverging.push_start_node(root_vid);

                debug!(
                    "calculate_diverging_fallback: root_vid={:?} reaches {:?}",
                    root_vid,
                    graph::depth_first_search(&coercion_graph, root_vid).collect::<Vec<_>>()
                );

                // drain the iterator to visit all nodes reachable from this node
                roots_reachable_from_diverging.complete_search();
            } else {
                non_diverging_vids.push(unsolved_vid);
            }
        }

        debug!(
            "calculate_diverging_fallback: roots_reachable_from_diverging={:?}",
            roots_reachable_from_diverging,
        );

        // Find all type variables N0 that are not reachable from a
        // diverging variable, and then compute the set reachable from
        // N0, which we call N. These are the *non-diverging* type
        // variables. (Note that this set consists of "root variables".)
        let mut roots_reachable_from_non_diverging = DepthFirstSearch::new(&coercion_graph);
        for &non_diverging_vid in &non_diverging_vids {
            let root_vid = self.root_var(non_diverging_vid);
            if roots_reachable_from_diverging.visited(root_vid) {
                continue;
            }
            roots_reachable_from_non_diverging.push_start_node(root_vid);
            roots_reachable_from_non_diverging.complete_search();
        }
        debug!(
            "calculate_diverging_fallback: roots_reachable_from_non_diverging={:?}",
            roots_reachable_from_non_diverging,
        );

        debug!("obligations: {:#?}", self.fulfillment_cx.borrow_mut().pending_obligations());

        // For each diverging variable, figure out whether it can
        // reach a member of N. If so, it falls back to `()`. Else
        // `!`.
        let mut diverging_fallback = UnordMap::with_capacity(diverging_vids.len());
        let unsafe_infer_vars = OnceCell::new();

        self.lint_obligations_broken_by_never_type_fallback_change(
            behavior,
            &diverging_vids,
            &coercion_graph,
        );

        for &diverging_vid in &diverging_vids {
            let diverging_ty = Ty::new_var(self.tcx, diverging_vid);
            let root_vid = self.root_var(diverging_vid);
            let can_reach_non_diverging = graph::depth_first_search(&coercion_graph, root_vid)
                .any(|n| roots_reachable_from_non_diverging.visited(n));

            let infer_var_infos: UnordBag<_> = self
                .infer_var_info
                .borrow()
                .items()
                .filter(|&(vid, _)| self.infcx.root_var(*vid) == root_vid)
                .map(|(_, info)| *info)
                .collect();

            let found_infer_var_info = InferVarInfo {
                self_in_trait: infer_var_infos.items().any(|info| info.self_in_trait),
                output: infer_var_infos.items().any(|info| info.output),
            };

            let mut fallback_to = |ty| {
                self.lint_never_type_fallback_flowing_into_unsafe_code(
                    &unsafe_infer_vars,
                    &coercion_graph,
                    root_vid,
                );

                diverging_fallback.insert(diverging_ty, ty);
            };

            match behavior {
                DivergingFallbackBehavior::ToUnit => {
                    debug!("fallback to () - legacy: {:?}", diverging_vid);
                    fallback_to(self.tcx.types.unit);
                }
                DivergingFallbackBehavior::ContextDependent => {
                    if found_infer_var_info.self_in_trait && found_infer_var_info.output {
                        // This case falls back to () to ensure that the code pattern in
                        // tests/ui/never_type/fallback-closure-ret.rs continues to
                        // compile when never_type_fallback is enabled.
                        //
                        // This rule is not readily explainable from first principles,
                        // but is rather intended as a patchwork fix to ensure code
                        // which compiles before the stabilization of never type
                        // fallback continues to work.
                        //
                        // Typically this pattern is encountered in a function taking a
                        // closure as a parameter, where the return type of that closure
                        // (checked by `relationship.output`) is expected to implement
                        // some trait (checked by `relationship.self_in_trait`). This
                        // can come up in non-closure cases too, so we do not limit this
                        // rule to specifically `FnOnce`.
                        //
                        // When the closure's body is something like `panic!()`, the
                        // return type would normally be inferred to `!`. However, it
                        // needs to fall back to `()` in order to still compile, as the
                        // trait is specifically implemented for `()` but not `!`.
                        //
                        // For details on the requirements for these relationships to be
                        // set, see the relationship finding module in
                        // compiler/rustc_trait_selection/src/traits/relationships.rs.
                        debug!("fallback to () - found trait and projection: {:?}", diverging_vid);
                        fallback_to(self.tcx.types.unit);
                    } else if can_reach_non_diverging {
                        debug!("fallback to () - reached non-diverging: {:?}", diverging_vid);
                        fallback_to(self.tcx.types.unit);
                    } else {
                        debug!("fallback to ! - all diverging: {:?}", diverging_vid);
                        fallback_to(self.tcx.types.never);
                    }
                }
                DivergingFallbackBehavior::ToNever => {
                    debug!(
                        "fallback to ! - `rustc_never_type_mode = \"fallback_to_never\")`: {:?}",
                        diverging_vid
                    );
                    fallback_to(self.tcx.types.never);
                }
                DivergingFallbackBehavior::NoFallback => {
                    debug!(
                        "no fallback - `rustc_never_type_mode = \"no_fallback\"`: {:?}",
                        diverging_vid
                    );
                }
            }
        }

        diverging_fallback
    }

    fn lint_never_type_fallback_flowing_into_unsafe_code(
        &self,
        unsafe_infer_vars: &OnceCell<UnordMap<ty::TyVid, (HirId, Span, UnsafeUseReason)>>,
        coercion_graph: &VecGraph<ty::TyVid, true>,
        root_vid: ty::TyVid,
    ) {
        let unsafe_infer_vars = unsafe_infer_vars.get_or_init(|| {
            let unsafe_infer_vars = compute_unsafe_infer_vars(self, self.body_id);
            debug!(?unsafe_infer_vars);
            unsafe_infer_vars
        });

        let affected_unsafe_infer_vars =
            graph::depth_first_search_as_undirected(&coercion_graph, root_vid)
                .filter_map(|x| unsafe_infer_vars.get(&x).copied())
                .collect::<Vec<_>>();

        let sugg = self.try_to_suggest_annotations(&[root_vid], coercion_graph);

        for (hir_id, span, reason) in affected_unsafe_infer_vars {
            self.tcx.emit_node_span_lint(
                lint::builtin::NEVER_TYPE_FALLBACK_FLOWING_INTO_UNSAFE,
                hir_id,
                span,
                match reason {
                    UnsafeUseReason::Call => {
                        errors::NeverTypeFallbackFlowingIntoUnsafe::Call { sugg: sugg.clone() }
                    }
                    UnsafeUseReason::Method => {
                        errors::NeverTypeFallbackFlowingIntoUnsafe::Method { sugg: sugg.clone() }
                    }
                    UnsafeUseReason::Path => {
                        errors::NeverTypeFallbackFlowingIntoUnsafe::Path { sugg: sugg.clone() }
                    }
                    UnsafeUseReason::UnionField => {
                        errors::NeverTypeFallbackFlowingIntoUnsafe::UnionField {
                            sugg: sugg.clone(),
                        }
                    }
                    UnsafeUseReason::Deref => {
                        errors::NeverTypeFallbackFlowingIntoUnsafe::Deref { sugg: sugg.clone() }
                    }
                },
            );
        }
    }

    fn lint_obligations_broken_by_never_type_fallback_change(
        &self,
        behavior: DivergingFallbackBehavior,
        diverging_vids: &[ty::TyVid],
        coercions: &VecGraph<ty::TyVid, true>,
    ) {
        let DivergingFallbackBehavior::ToUnit = behavior else { return };

        // Fallback happens if and only if there are diverging variables
        if diverging_vids.is_empty() {
            return;
        }

        // Returns errors which happen if fallback is set to `fallback`
        let remaining_errors_if_fallback_to = |fallback| {
            self.probe(|_| {
                let obligations = self.fulfillment_cx.borrow().pending_obligations();
                let ocx = ObligationCtxt::new_with_diagnostics(&self.infcx);
                ocx.register_obligations(obligations.iter().cloned());

                for &diverging_vid in diverging_vids {
                    let diverging_ty = Ty::new_var(self.tcx, diverging_vid);

                    ocx.eq(&ObligationCause::dummy(), self.param_env, diverging_ty, fallback)
                        .expect("expected diverging var to be unconstrained");
                }

                ocx.select_where_possible()
            })
        };

        // If we have no errors with `fallback = ()`, but *do* have errors with `fallback = !`,
        // then this code will be broken by the never type fallback change.
        let unit_errors = remaining_errors_if_fallback_to(self.tcx.types.unit);
        if unit_errors.is_empty()
            && let mut never_errors = remaining_errors_if_fallback_to(self.tcx.types.never)
            && let [never_error, ..] = never_errors.as_mut_slice()
        {
            self.adjust_fulfillment_error_for_expr_obligation(never_error);
            let sugg = self.try_to_suggest_annotations(diverging_vids, coercions);
            self.tcx.emit_node_span_lint(
                lint::builtin::DEPENDENCY_ON_UNIT_NEVER_TYPE_FALLBACK,
                self.tcx.local_def_id_to_hir_id(self.body_id),
                self.tcx.def_span(self.body_id),
                errors::DependencyOnUnitNeverTypeFallback {
                    obligation_span: never_error.obligation.cause.span,
                    obligation: never_error.obligation.predicate,
                    sugg,
                },
            )
        }
    }

    /// Returns a graph whose nodes are (unresolved) inference variables and where
    /// an edge `?A -> ?B` indicates that the variable `?A` is coerced to `?B`.
    fn create_coercion_graph(&self) -> VecGraph<ty::TyVid, true> {
        let pending_obligations = self.fulfillment_cx.borrow_mut().pending_obligations();
        debug!("create_coercion_graph: pending_obligations={:?}", pending_obligations);
        let coercion_edges: Vec<(ty::TyVid, ty::TyVid)> = pending_obligations
            .into_iter()
            .filter_map(|obligation| {
                // The predicates we are looking for look like `Coerce(?A -> ?B)`.
                // They will have no bound variables.
                obligation.predicate.kind().no_bound_vars()
            })
            .filter_map(|atom| {
                // We consider both subtyping and coercion to imply 'flow' from
                // some position in the code `a` to a different position `b`.
                // This is then used to determine which variables interact with
                // live code, and as such must fall back to `()` to preserve
                // soundness.
                //
                // In practice currently the two ways that this happens is
                // coercion and subtyping.
                let (a, b) = match atom {
                    ty::PredicateKind::Coerce(ty::CoercePredicate { a, b }) => (a, b),
                    ty::PredicateKind::Subtype(ty::SubtypePredicate { a_is_expected: _, a, b }) => {
                        (a, b)
                    }
                    _ => return None,
                };

                let a_vid = self.root_vid(a)?;
                let b_vid = self.root_vid(b)?;
                Some((a_vid, b_vid))
            })
            .collect();
        debug!("create_coercion_graph: coercion_edges={:?}", coercion_edges);
        let num_ty_vars = self.num_ty_vars();

        VecGraph::new(num_ty_vars, coercion_edges)
    }

    /// If `ty` is an unresolved type variable, returns its root vid.
    fn root_vid(&self, ty: Ty<'tcx>) -> Option<ty::TyVid> {
        Some(self.root_var(self.shallow_resolve(ty).ty_vid()?))
    }

    /// Given a set of diverging vids and coercions, walk the HIR to gather a
    /// set of suggestions which can be applied to preserve fallback to unit.
    fn try_to_suggest_annotations(
        &self,
        diverging_vids: &[ty::TyVid],
        coercions: &VecGraph<ty::TyVid, true>,
    ) -> errors::SuggestAnnotations {
        let body =
            self.tcx.hir_maybe_body_owned_by(self.body_id).expect("body id must have an owner");
        // For each diverging var, look through the HIR for a place to give it
        // a type annotation. We do this per var because we only really need one
        // suggestion to influence a var to be `()`.
        let suggestions = diverging_vids
            .iter()
            .copied()
            .filter_map(|vid| {
                let reachable_vids =
                    graph::depth_first_search_as_undirected(coercions, vid).collect();
                AnnotateUnitFallbackVisitor { reachable_vids, fcx: self }
                    .visit_expr(body.value)
                    .break_value()
            })
            .collect();
        errors::SuggestAnnotations { suggestions }
    }
}

/// Try to walk the HIR to find a place to insert a useful suggestion
/// to preserve fallback to `()` in 2024.
struct AnnotateUnitFallbackVisitor<'a, 'tcx> {
    reachable_vids: FxHashSet<ty::TyVid>,
    fcx: &'a FnCtxt<'a, 'tcx>,
}
impl<'tcx> AnnotateUnitFallbackVisitor<'_, 'tcx> {
    // For a given path segment, if it's missing a turbofish, try to suggest adding
    // one so we can constrain an argument to `()`. To keep the suggestion simple,
    // we want to simply suggest `_` for all the other args. This (for now) only
    // works when there are only type variables (and region variables, since we can
    // elide them)...
    fn suggest_for_segment(
        &self,
        arg_segment: &'tcx hir::PathSegment<'tcx>,
        def_id: DefId,
        id: HirId,
    ) -> ControlFlow<errors::SuggestAnnotation> {
        if arg_segment.args.is_none()
            && let Some(all_args) = self.fcx.typeck_results.borrow().node_args_opt(id)
            && let generics = self.fcx.tcx.generics_of(def_id)
            && let args = all_args[generics.parent_count..].iter().zip(&generics.own_params)
            // We can't turbofish consts :(
            && args.clone().all(|(_, param)| matches!(param.kind, ty::GenericParamDefKind::Type { .. } | ty::GenericParamDefKind::Lifetime))
        {
            // We filter out APITs, which are not turbofished.
            let non_apit_type_args = args.filter(|(_, param)| {
                matches!(param.kind, ty::GenericParamDefKind::Type { synthetic: false, .. })
            });
            let n_tys = non_apit_type_args.clone().count();
            for (idx, (arg, _)) in non_apit_type_args.enumerate() {
                if let Some(ty) = arg.as_type()
                    && let Some(vid) = self.fcx.root_vid(ty)
                    && self.reachable_vids.contains(&vid)
                {
                    return ControlFlow::Break(errors::SuggestAnnotation::Turbo(
                        arg_segment.ident.span.shrink_to_hi(),
                        n_tys,
                        idx,
                    ));
                }
            }
        }
        ControlFlow::Continue(())
    }
}
impl<'tcx> Visitor<'tcx> for AnnotateUnitFallbackVisitor<'_, 'tcx> {
    type Result = ControlFlow<errors::SuggestAnnotation>;

    fn visit_infer(
        &mut self,
        inf_id: HirId,
        inf_span: Span,
        _kind: InferKind<'tcx>,
    ) -> Self::Result {
        // Try to replace `_` with `()`.
        if let Some(ty) = self.fcx.typeck_results.borrow().node_type_opt(inf_id)
            && let Some(vid) = self.fcx.root_vid(ty)
            && self.reachable_vids.contains(&vid)
            && inf_span.can_be_used_for_suggestions()
        {
            return ControlFlow::Break(errors::SuggestAnnotation::Unit(inf_span));
        }

        ControlFlow::Continue(())
    }

    fn visit_qpath(
        &mut self,
        qpath: &'tcx rustc_hir::QPath<'tcx>,
        id: HirId,
        span: Span,
    ) -> Self::Result {
        let arg_segment = match qpath {
            hir::QPath::Resolved(_, path) => {
                path.segments.last().expect("paths should have a segment")
            }
            hir::QPath::TypeRelative(_, segment) => segment,
            hir::QPath::LangItem(..) => {
                return hir::intravisit::walk_qpath(self, qpath, id);
            }
        };
        // Alternatively, try to turbofish `::<_, (), _>`.
        if let Some(def_id) = self.fcx.typeck_results.borrow().qpath_res(qpath, id).opt_def_id()
            && span.can_be_used_for_suggestions()
        {
            self.suggest_for_segment(arg_segment, def_id, id)?;
        }
        hir::intravisit::walk_qpath(self, qpath, id)
    }

    fn visit_expr(&mut self, expr: &'tcx hir::Expr<'tcx>) -> Self::Result {
        if let hir::ExprKind::Closure(&hir::Closure { body, .. })
        | hir::ExprKind::ConstBlock(hir::ConstBlock { body, .. }) = expr.kind
        {
            self.visit_body(self.fcx.tcx.hir_body(body))?;
        }

        // Try to suggest adding an explicit qself `()` to a trait method path.
        // i.e. changing `Default::default()` to `<() as Default>::default()`.
        if let hir::ExprKind::Path(hir::QPath::Resolved(None, path)) = expr.kind
            && let Res::Def(DefKind::AssocFn, def_id) = path.res
            && self.fcx.tcx.trait_of_assoc(def_id).is_some()
            && let Some(args) = self.fcx.typeck_results.borrow().node_args_opt(expr.hir_id)
            && let self_ty = args.type_at(0)
            && let Some(vid) = self.fcx.root_vid(self_ty)
            && self.reachable_vids.contains(&vid)
            && let [.., trait_segment, _method_segment] = path.segments
            && expr.span.can_be_used_for_suggestions()
        {
            let span = path.span.shrink_to_lo().to(trait_segment.ident.span);
            return ControlFlow::Break(errors::SuggestAnnotation::Path(span));
        }

        // Or else, try suggesting turbofishing the method args.
        if let hir::ExprKind::MethodCall(segment, ..) = expr.kind
            && let Some(def_id) =
                self.fcx.typeck_results.borrow().type_dependent_def_id(expr.hir_id)
            && expr.span.can_be_used_for_suggestions()
        {
            self.suggest_for_segment(segment, def_id, expr.hir_id)?;
        }

        hir::intravisit::walk_expr(self, expr)
    }

    fn visit_local(&mut self, local: &'tcx hir::LetStmt<'tcx>) -> Self::Result {
        // For a local, try suggest annotating the type if it's missing.
        if let hir::LocalSource::Normal = local.source
            && let None = local.ty
            && let Some(ty) = self.fcx.typeck_results.borrow().node_type_opt(local.hir_id)
            && let Some(vid) = self.fcx.root_vid(ty)
            && self.reachable_vids.contains(&vid)
            && local.span.can_be_used_for_suggestions()
        {
            return ControlFlow::Break(errors::SuggestAnnotation::Local(
                local.pat.span.shrink_to_hi(),
            ));
        }
        hir::intravisit::walk_local(self, local)
    }
}

#[derive(Debug, Copy, Clone)]
pub(crate) enum UnsafeUseReason {
    Call,
    Method,
    Path,
    UnionField,
    Deref,
}

/// Finds all type variables which are passed to an `unsafe` operation.
///
/// For example, for this function `f`:
/// ```ignore (demonstrative)
/// fn f() {
///     unsafe {
///         let x /* ?X */ = core::mem::zeroed();
///         //               ^^^^^^^^^^^^^^^^^^^ -- hir_id, span, reason
///
///         let y = core::mem::zeroed::<Option<_ /* ?Y */>>();
///         //      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ -- hir_id, span, reason
///     }
/// }
/// ```
///
/// `compute_unsafe_infer_vars` will return `{ id(?X) -> (hir_id, span, Call) }`
fn compute_unsafe_infer_vars<'a, 'tcx>(
    fcx: &'a FnCtxt<'a, 'tcx>,
    body_id: LocalDefId,
) -> UnordMap<ty::TyVid, (HirId, Span, UnsafeUseReason)> {
    let body = fcx.tcx.hir_maybe_body_owned_by(body_id).expect("body id must have an owner");
    let mut res = UnordMap::default();

    struct UnsafeInferVarsVisitor<'a, 'tcx> {
        fcx: &'a FnCtxt<'a, 'tcx>,
        res: &'a mut UnordMap<ty::TyVid, (HirId, Span, UnsafeUseReason)>,
    }

    impl Visitor<'_> for UnsafeInferVarsVisitor<'_, '_> {
        fn visit_expr(&mut self, ex: &'_ hir::Expr<'_>) {
            let typeck_results = self.fcx.typeck_results.borrow();

            match ex.kind {
                hir::ExprKind::MethodCall(..) => {
                    if let Some(def_id) = typeck_results.type_dependent_def_id(ex.hir_id)
                        && let method_ty = self.fcx.tcx.type_of(def_id).instantiate_identity()
                        && let sig = method_ty.fn_sig(self.fcx.tcx)
                        && sig.safety().is_unsafe()
                    {
                        let mut collector = InferVarCollector {
                            value: (ex.hir_id, ex.span, UnsafeUseReason::Method),
                            res: self.res,
                        };

                        // Collect generic arguments (incl. `Self`) of the method
                        typeck_results
                            .node_args(ex.hir_id)
                            .types()
                            .for_each(|t| t.visit_with(&mut collector));
                    }
                }

                hir::ExprKind::Call(func, ..) => {
                    let func_ty = typeck_results.expr_ty(func);

                    if func_ty.is_fn()
                        && let sig = func_ty.fn_sig(self.fcx.tcx)
                        && sig.safety().is_unsafe()
                    {
                        let mut collector = InferVarCollector {
                            value: (ex.hir_id, ex.span, UnsafeUseReason::Call),
                            res: self.res,
                        };

                        // Try collecting generic arguments of the function.
                        // Note that we do this below for any paths (that don't have to be called),
                        // but there we do it with a different span/reason.
                        // This takes priority.
                        typeck_results
                            .node_args(func.hir_id)
                            .types()
                            .for_each(|t| t.visit_with(&mut collector));

                        // Also check the return type, for cases like `returns_unsafe_fn_ptr()()`
                        sig.output().visit_with(&mut collector);
                    }
                }

                // Check paths which refer to functions.
                // We do this, instead of only checking `Call` to make sure the lint can't be
                // avoided by storing unsafe function in a variable.
                hir::ExprKind::Path(_) => {
                    let ty = typeck_results.expr_ty(ex);

                    // If this path refers to an unsafe function, collect inference variables which may affect it.
                    // `is_fn` excludes closures, but those can't be unsafe.
                    if ty.is_fn()
                        && let sig = ty.fn_sig(self.fcx.tcx)
                        && sig.safety().is_unsafe()
                    {
                        let mut collector = InferVarCollector {
                            value: (ex.hir_id, ex.span, UnsafeUseReason::Path),
                            res: self.res,
                        };

                        // Collect generic arguments of the function
                        typeck_results
                            .node_args(ex.hir_id)
                            .types()
                            .for_each(|t| t.visit_with(&mut collector));
                    }
                }

                hir::ExprKind::Unary(hir::UnOp::Deref, pointer) => {
                    if let ty::RawPtr(pointee, _) = typeck_results.expr_ty(pointer).kind() {
                        pointee.visit_with(&mut InferVarCollector {
                            value: (ex.hir_id, ex.span, UnsafeUseReason::Deref),
                            res: self.res,
                        });
                    }
                }

                hir::ExprKind::Field(base, _) => {
                    let base_ty = typeck_results.expr_ty(base);

                    if base_ty.is_union() {
                        typeck_results.expr_ty(ex).visit_with(&mut InferVarCollector {
                            value: (ex.hir_id, ex.span, UnsafeUseReason::UnionField),
                            res: self.res,
                        });
                    }
                }

                _ => (),
            };

            hir::intravisit::walk_expr(self, ex);
        }
    }

    struct InferVarCollector<'r, V> {
        value: V,
        res: &'r mut UnordMap<ty::TyVid, V>,
    }

    impl<'tcx, V: Copy> ty::TypeVisitor<TyCtxt<'tcx>> for InferVarCollector<'_, V> {
        fn visit_ty(&mut self, t: Ty<'tcx>) {
            if let Some(vid) = t.ty_vid() {
                _ = self.res.try_insert(vid, self.value);
            } else {
                t.super_visit_with(self)
            }
        }
    }

    UnsafeInferVarsVisitor { fcx, res: &mut res }.visit_expr(&body.value);

    debug!(?res, "collected the following unsafe vars for {body_id:?}");

    res
}
