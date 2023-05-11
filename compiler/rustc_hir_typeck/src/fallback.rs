use crate::FnCtxt;
use rustc_data_structures::{
    fx::{FxHashMap, FxHashSet},
    graph::WithSuccessors,
    graph::{iterate::DepthFirstSearch, vec_graph::VecGraph},
};
use rustc_middle::ty::{self, Ty};

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

        // Check if we have any unsolved variables. If not, no need for fallback.
        let unsolved_variables = self.unsolved_variables();
        if unsolved_variables.is_empty() {
            return;
        }

        let diverging_fallback = self.calculate_diverging_fallback(&unsolved_variables);

        // We do fallback in two passes, to try to generate
        // better error messages.
        // The first time, we do *not* replace opaque types.
        for ty in unsolved_variables {
            debug!("unsolved_variable = {:?}", ty);
            self.fallback_if_possible(ty, &diverging_fallback);
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
        diverging_fallback: &FxHashMap<Ty<'tcx>, Ty<'tcx>>,
    ) {
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
            _ if let Some(e) = self.tainted_by_errors() => self.tcx.ty_error(e),
            ty::Infer(ty::IntVar(_)) => self.tcx.types.i32,
            ty::Infer(ty::FloatVar(_)) => self.tcx.types.f64,
            _ => match diverging_fallback.get(&ty) {
                Some(&fallback_ty) => fallback_ty,
                None => return,
            },
        };
        debug!("fallback_if_possible(ty={:?}): defaulting to `{:?}`", ty, fallback);

        let span = self
            .infcx
            .type_var_origin(ty)
            .map(|origin| origin.span)
            .unwrap_or(rustc_span::DUMMY_SP);
        self.demand_eqtype(span, ty, fallback);
        self.fallback_has_occurred.set(true);
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
        unsolved_variables: &[Ty<'tcx>],
    ) -> FxHashMap<Ty<'tcx>, Ty<'tcx>> {
        debug!("calculate_diverging_fallback({:?})", unsolved_variables);

        // Construct a coercion graph where an edge `A -> B` indicates
        // a type variable is that is coerced
        let coercion_graph = self.create_coercion_graph();

        // Extract the unsolved type inference variable vids; note that some
        // unsolved variables are integer/float variables and are excluded.
        let unsolved_vids = unsolved_variables.iter().filter_map(|ty| ty.ty_vid());

        // Compute the diverging root vids D -- that is, the root vid of
        // those type variables that (a) are the target of a coercion from
        // a `!` type and (b) have not yet been solved.
        //
        // These variables are the ones that are targets for fallback to
        // either `!` or `()`.
        let diverging_roots: FxHashSet<ty::TyVid> = self
            .diverging_type_vars
            .borrow()
            .iter()
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
                    coercion_graph.depth_first_search(root_vid).collect::<Vec<_>>()
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
        let mut diverging_fallback = FxHashMap::default();
        diverging_fallback.reserve(diverging_vids.len());
        for &diverging_vid in &diverging_vids {
            let diverging_ty = self.tcx.mk_ty_var(diverging_vid);
            let root_vid = self.root_var(diverging_vid);
            let can_reach_non_diverging = coercion_graph
                .depth_first_search(root_vid)
                .any(|n| roots_reachable_from_non_diverging.visited(n));

            let mut found_infer_var_info = ty::InferVarInfo { self_in_trait: false, output: false };

            for (vid, info) in self.inh.infer_var_info.borrow().iter() {
                if self.infcx.root_var(*vid) == root_vid {
                    found_infer_var_info.self_in_trait |= info.self_in_trait;
                    found_infer_var_info.output |= info.output;
                }
            }

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
                diverging_fallback.insert(diverging_ty, self.tcx.types.unit);
            } else if can_reach_non_diverging {
                debug!("fallback to () - reached non-diverging: {:?}", diverging_vid);
                diverging_fallback.insert(diverging_ty, self.tcx.types.unit);
            } else {
                debug!("fallback to ! - all diverging: {:?}", diverging_vid);
                diverging_fallback.insert(diverging_ty, self.tcx.mk_diverging_default());
            }
        }

        diverging_fallback
    }

    /// Returns a graph whose nodes are (unresolved) inference variables and where
    /// an edge `?A -> ?B` indicates that the variable `?A` is coerced to `?B`.
    fn create_coercion_graph(&self) -> VecGraph<ty::TyVid> {
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
                let (a, b) = if let ty::PredicateKind::Coerce(ty::CoercePredicate { a, b }) = atom {
                    (a, b)
                } else if let ty::PredicateKind::Subtype(ty::SubtypePredicate {
                    a_is_expected: _,
                    a,
                    b,
                }) = atom
                {
                    (a, b)
                } else {
                    return None;
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
}
