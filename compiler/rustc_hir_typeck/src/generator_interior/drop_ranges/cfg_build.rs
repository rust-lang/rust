use super::{
    for_each_consumable, record_consumed_borrow::ConsumedAndBorrowedPlaces, DropRangesBuilder,
    NodeInfo, PostOrderId, TrackedValue, TrackedValueIndex,
};
use hir::{
    intravisit::{self, Visitor},
    Body, Expr, ExprKind, Guard, HirId, LoopIdError,
};
use rustc_data_structures::fx::{FxHashMap, FxHashSet};
use rustc_hir as hir;
use rustc_index::IndexVec;
use rustc_infer::infer::InferCtxt;
use rustc_middle::{
    hir::map::Map,
    ty::{ParamEnv, TyCtxt, TypeVisitableExt, TypeckResults},
};
use std::mem::swap;

/// Traverses the body to find the control flow graph and locations for the
/// relevant places are dropped or reinitialized.
///
/// The resulting structure still needs to be iterated to a fixed point, which
/// can be done with propagate_to_fixpoint in cfg_propagate.
pub(super) fn build_control_flow_graph<'tcx>(
    infcx: &InferCtxt<'tcx>,
    typeck_results: &TypeckResults<'tcx>,
    param_env: ParamEnv<'tcx>,
    consumed_borrowed_places: ConsumedAndBorrowedPlaces,
    body: &'tcx Body<'tcx>,
    num_exprs: usize,
) -> (DropRangesBuilder, FxHashSet<HirId>) {
    let mut drop_range_visitor = DropRangeVisitor::new(
        infcx,
        typeck_results,
        param_env,
        consumed_borrowed_places,
        num_exprs,
    );
    intravisit::walk_body(&mut drop_range_visitor, body);

    drop_range_visitor.drop_ranges.process_deferred_edges();
    if let Some(filename) = &infcx.tcx.sess.opts.unstable_opts.dump_drop_tracking_cfg {
        super::cfg_visualize::write_graph_to_file(
            &drop_range_visitor.drop_ranges,
            filename,
            infcx.tcx,
        );
    }

    (drop_range_visitor.drop_ranges, drop_range_visitor.places.borrowed_temporaries)
}

/// This struct is used to gather the information for `DropRanges` to determine the regions of the
/// HIR tree for which a value is dropped.
///
/// We are interested in points where a variables is dropped or initialized, and the control flow
/// of the code. We identify locations in code by their post-order traversal index, so it is
/// important for this traversal to match that in `RegionResolutionVisitor` and `InteriorVisitor`.
///
/// We make several simplifying assumptions, with the goal of being more conservative than
/// necessary rather than less conservative (since being less conservative is unsound, but more
/// conservative is still safe). These assumptions are:
///
/// 1. Moving a variable `a` counts as a move of the whole variable.
/// 2. Moving a partial path like `a.b.c` is ignored.
/// 3. Reinitializing through a field (e.g. `a.b.c = 5`) counts as a reinitialization of all of
///    `a`.
///
/// Some examples:
///
/// Rule 1:
/// ```rust
/// let mut a = (vec![0], vec![0]);
/// drop(a);
/// // `a` is not considered initialized.
/// ```
///
/// Rule 2:
/// ```rust
/// let mut a = (vec![0], vec![0]);
/// drop(a.0);
/// drop(a.1);
/// // `a` is still considered initialized.
/// ```
///
/// Rule 3:
/// ```compile_fail,E0382
/// let mut a = (vec![0], vec![0]);
/// drop(a);
/// a.1 = vec![1];
/// // all of `a` is considered initialized
/// ```

struct DropRangeVisitor<'a, 'tcx> {
    typeck_results: &'a TypeckResults<'tcx>,
    infcx: &'a InferCtxt<'tcx>,
    param_env: ParamEnv<'tcx>,
    places: ConsumedAndBorrowedPlaces,
    drop_ranges: DropRangesBuilder,
    expr_index: PostOrderId,
    label_stack: Vec<(Option<rustc_ast::Label>, PostOrderId)>,
}

impl<'a, 'tcx> DropRangeVisitor<'a, 'tcx> {
    fn new(
        infcx: &'a InferCtxt<'tcx>,
        typeck_results: &'a TypeckResults<'tcx>,
        param_env: ParamEnv<'tcx>,
        places: ConsumedAndBorrowedPlaces,
        num_exprs: usize,
    ) -> Self {
        debug!("consumed_places: {:?}", places.consumed);
        let drop_ranges = DropRangesBuilder::new(
            places.consumed.iter().flat_map(|(_, places)| places.iter().cloned()),
            infcx.tcx.hir(),
            num_exprs,
        );
        Self {
            infcx,
            typeck_results,
            param_env,
            places,
            drop_ranges,
            expr_index: PostOrderId::from_u32(0),
            label_stack: vec![],
        }
    }

    fn tcx(&self) -> TyCtxt<'tcx> {
        self.infcx.tcx
    }

    fn record_drop(&mut self, value: TrackedValue) {
        if self.places.borrowed.contains(&value) {
            debug!("not marking {:?} as dropped because it is borrowed at some point", value);
        } else {
            debug!("marking {:?} as dropped at {:?}", value, self.expr_index);
            let count = self.expr_index;
            self.drop_ranges.drop_at(value, count);
        }
    }

    /// ExprUseVisitor's consume callback doesn't go deep enough for our purposes in all
    /// expressions. This method consumes a little deeper into the expression when needed.
    fn consume_expr(&mut self, expr: &hir::Expr<'_>) {
        debug!("consuming expr {:?}, count={:?}", expr.kind, self.expr_index);
        let places = self
            .places
            .consumed
            .get(&expr.hir_id)
            .map_or(vec![], |places| places.iter().cloned().collect());
        for place in places {
            trace!(?place, "consuming place");
            for_each_consumable(self.tcx().hir(), place, |value| self.record_drop(value));
        }
    }

    /// Marks an expression as being reinitialized.
    ///
    /// Note that we always approximated on the side of things being more
    /// initialized than they actually are, as opposed to less. In cases such
    /// as `x.y = ...`, we would consider all of `x` as being initialized
    /// instead of just the `y` field.
    ///
    /// This is because it is always safe to consider something initialized
    /// even when it is not, but the other way around will cause problems.
    ///
    /// In the future, we will hopefully tighten up these rules to be more
    /// precise.
    fn reinit_expr(&mut self, expr: &hir::Expr<'_>) {
        // Walk the expression to find the base. For example, in an expression
        // like `*a[i].x`, we want to find the `a` and mark that as
        // reinitialized.
        match expr.kind {
            ExprKind::Path(hir::QPath::Resolved(
                _,
                hir::Path { res: hir::def::Res::Local(hir_id), .. },
            )) => {
                // This is the base case, where we have found an actual named variable.

                let location = self.expr_index;
                debug!("reinitializing {:?} at {:?}", hir_id, location);
                self.drop_ranges.reinit_at(TrackedValue::Variable(*hir_id), location);
            }

            ExprKind::Field(base, _) => self.reinit_expr(base),

            // Most expressions do not refer to something where we need to track
            // reinitializations.
            //
            // Some of these may be interesting in the future
            ExprKind::Path(..)
            | ExprKind::ConstBlock(..)
            | ExprKind::Array(..)
            | ExprKind::Call(..)
            | ExprKind::MethodCall(..)
            | ExprKind::Tup(..)
            | ExprKind::Binary(..)
            | ExprKind::Unary(..)
            | ExprKind::Lit(..)
            | ExprKind::Cast(..)
            | ExprKind::Type(..)
            | ExprKind::DropTemps(..)
            | ExprKind::Let(..)
            | ExprKind::If(..)
            | ExprKind::Loop(..)
            | ExprKind::Match(..)
            | ExprKind::Closure { .. }
            | ExprKind::Block(..)
            | ExprKind::Assign(..)
            | ExprKind::AssignOp(..)
            | ExprKind::Index(..)
            | ExprKind::AddrOf(..)
            | ExprKind::Break(..)
            | ExprKind::Continue(..)
            | ExprKind::Ret(..)
            | ExprKind::Become(..)
            | ExprKind::InlineAsm(..)
            | ExprKind::OffsetOf(..)
            | ExprKind::Struct(..)
            | ExprKind::Repeat(..)
            | ExprKind::Yield(..)
            | ExprKind::Err(_) => (),
        }
    }

    /// For an expression with an uninhabited return type (e.g. a function that returns !),
    /// this adds a self edge to the CFG to model the fact that the function does not
    /// return.
    fn handle_uninhabited_return(&mut self, expr: &Expr<'tcx>) {
        let ty = self.typeck_results.expr_ty(expr);
        let ty = self.infcx.resolve_vars_if_possible(ty);
        if ty.has_non_region_infer() {
            self.tcx()
                .sess
                .delay_span_bug(expr.span, format!("could not resolve infer vars in `{ty}`"));
            return;
        }
        let ty = self.tcx().erase_regions(ty);
        let m = self.tcx().parent_module(expr.hir_id).to_def_id();
        if !ty.is_inhabited_from(self.tcx(), m, self.param_env) {
            // This function will not return. We model this fact as an infinite loop.
            self.drop_ranges.add_control_edge(self.expr_index + 1, self.expr_index + 1);
        }
    }

    /// Map a Destination to an equivalent expression node
    ///
    /// The destination field of a Break or Continue expression can target either an
    /// expression or a block. The drop range analysis, however, only deals in
    /// expression nodes, so blocks that might be the destination of a Break or Continue
    /// will not have a PostOrderId.
    ///
    /// If the destination is an expression, this function will simply return that expression's
    /// hir_id. If the destination is a block, this function will return the hir_id of last
    /// expression in the block.
    fn find_target_expression_from_destination(
        &self,
        destination: hir::Destination,
    ) -> Result<HirId, LoopIdError> {
        destination.target_id.map(|target| {
            let node = self.tcx().hir().get(target);
            match node {
                hir::Node::Expr(_) => target,
                hir::Node::Block(b) => find_last_block_expression(b),
                hir::Node::Param(..)
                | hir::Node::Item(..)
                | hir::Node::ForeignItem(..)
                | hir::Node::TraitItem(..)
                | hir::Node::ImplItem(..)
                | hir::Node::Variant(..)
                | hir::Node::Field(..)
                | hir::Node::AnonConst(..)
                | hir::Node::ConstBlock(..)
                | hir::Node::Stmt(..)
                | hir::Node::PathSegment(..)
                | hir::Node::Ty(..)
                | hir::Node::TypeBinding(..)
                | hir::Node::TraitRef(..)
                | hir::Node::Pat(..)
                | hir::Node::PatField(..)
                | hir::Node::ExprField(..)
                | hir::Node::Arm(..)
                | hir::Node::Local(..)
                | hir::Node::Ctor(..)
                | hir::Node::Lifetime(..)
                | hir::Node::GenericParam(..)
                | hir::Node::Crate(..)
                | hir::Node::Infer(..) => bug!("Unsupported branch target: {:?}", node),
            }
        })
    }
}

fn find_last_block_expression(block: &hir::Block<'_>) -> HirId {
    block.expr.map_or_else(
        // If there is no tail expression, there will be at least one statement in the
        // block because the block contains a break or continue statement.
        || block.stmts.last().unwrap().hir_id,
        |expr| expr.hir_id,
    )
}

impl<'a, 'tcx> Visitor<'tcx> for DropRangeVisitor<'a, 'tcx> {
    fn visit_expr(&mut self, expr: &'tcx Expr<'tcx>) {
        let mut reinit = None;
        match expr.kind {
            ExprKind::Assign(lhs, rhs, _) => {
                self.visit_expr(rhs);
                self.visit_expr(lhs);

                reinit = Some(lhs);
            }

            ExprKind::If(test, if_true, if_false) => {
                self.visit_expr(test);

                let fork = self.expr_index;

                self.drop_ranges.add_control_edge(fork, self.expr_index + 1);
                self.visit_expr(if_true);
                let true_end = self.expr_index;

                self.drop_ranges.add_control_edge(fork, self.expr_index + 1);
                if let Some(if_false) = if_false {
                    self.visit_expr(if_false);
                }

                self.drop_ranges.add_control_edge(true_end, self.expr_index + 1);
            }
            ExprKind::Match(scrutinee, arms, ..) => {
                // We walk through the match expression almost like a chain of if expressions.
                // Here's a diagram to follow along with:
                //
                //           ┌─┐
                //     match │A│ {
                //       ┌───┴─┘
                //       │
                //      ┌▼┌───►┌─┐   ┌─┐
                //      │B│ if │C│ =>│D│,
                //      └─┘    ├─┴──►└─┴──────┐
                //          ┌──┘              │
                //       ┌──┘                 │
                //       │                    │
                //      ┌▼┌───►┌─┐   ┌─┐      │
                //      │E│ if │F│ =>│G│,     │
                //      └─┘    ├─┴──►└─┴┐     │
                //             │        │     │
                //     }       ▼        ▼     │
                //     ┌─┐◄───────────────────┘
                //     │H│
                //     └─┘
                //
                // The order we want is that the scrutinee (A) flows into the first pattern (B),
                // which flows into the guard (C). Then the guard either flows into the arm body
                // (D) or into the start of the next arm (E). Finally, the body flows to the end
                // of the match block (H).
                //
                // The subsequent arms follow the same ordering. First we go to the pattern, then
                // the guard (if present, otherwise it flows straight into the body), then into
                // the body and then to the end of the match expression.
                //
                // The comments below show which edge is being added.
                self.visit_expr(scrutinee);

                let (guard_exit, arm_end_ids) = arms.iter().fold(
                    (self.expr_index, vec![]),
                    |(incoming_edge, mut arm_end_ids), hir::Arm { pat, body, guard, .. }| {
                        // A -> B, or C -> E
                        self.drop_ranges.add_control_edge(incoming_edge, self.expr_index + 1);
                        self.visit_pat(pat);
                        // B -> C and E -> F are added implicitly due to the traversal order.
                        match guard {
                            Some(Guard::If(expr)) => self.visit_expr(expr),
                            Some(Guard::IfLet(let_expr)) => {
                                self.visit_let_expr(let_expr);
                            }
                            None => (),
                        }
                        // Likewise, C -> D and F -> G are added implicitly.

                        // Save C, F, so we can add the other outgoing edge.
                        let to_next_arm = self.expr_index;

                        // The default edge does not get added since we also have an explicit edge,
                        // so we also need to add an edge to the next node as well.
                        //
                        // This adds C -> D, F -> G
                        self.drop_ranges.add_control_edge(self.expr_index, self.expr_index + 1);
                        self.visit_expr(body);

                        // Save the end of the body so we can add the exit edge once we know where
                        // the exit is.
                        arm_end_ids.push(self.expr_index);

                        // Pass C to the next iteration, as well as vec![D]
                        //
                        // On the last round through, we pass F and vec![D, G] so that we can
                        // add all the exit edges.
                        (to_next_arm, arm_end_ids)
                    },
                );
                // F -> H
                self.drop_ranges.add_control_edge(guard_exit, self.expr_index + 1);

                arm_end_ids.into_iter().for_each(|arm_end| {
                    // D -> H, G -> H
                    self.drop_ranges.add_control_edge(arm_end, self.expr_index + 1)
                });
            }

            ExprKind::Loop(body, label, ..) => {
                let loop_begin = self.expr_index + 1;
                self.label_stack.push((label, loop_begin));
                if body.stmts.is_empty() && body.expr.is_none() {
                    // For empty loops we won't have updated self.expr_index after visiting the
                    // body, meaning we'd get an edge from expr_index to expr_index + 1, but
                    // instead we want an edge from expr_index + 1 to expr_index + 1.
                    self.drop_ranges.add_control_edge(loop_begin, loop_begin);
                } else {
                    self.visit_block(body);
                    self.drop_ranges.add_control_edge(self.expr_index, loop_begin);
                }
                self.label_stack.pop();
            }
            // Find the loop entry by searching through the label stack for either the last entry
            // (if label is none), or the first entry where the label matches this one. The Loop
            // case maintains this stack mapping labels to the PostOrderId for the loop entry.
            ExprKind::Continue(hir::Destination { label, .. }, ..) => self
                .label_stack
                .iter()
                .rev()
                .find(|(loop_label, _)| label.is_none() || *loop_label == label)
                .map_or((), |(_, target)| {
                    self.drop_ranges.add_control_edge(self.expr_index, *target)
                }),

            ExprKind::Break(destination, value) => {
                // destination either points to an expression or to a block. We use
                // find_target_expression_from_destination to use the last expression of the block
                // if destination points to a block.
                //
                // We add an edge to the hir_id of the expression/block we are breaking out of, and
                // then in process_deferred_edges we will map this hir_id to its PostOrderId, which
                // will refer to the end of the block due to the post order traversal.
                self.find_target_expression_from_destination(destination).map_or((), |target| {
                    self.drop_ranges.add_control_edge_hir_id(self.expr_index, target)
                });

                if let Some(value) = value {
                    self.visit_expr(value);
                }
            }

            ExprKind::Become(_call) => bug!("encountered a tail-call inside a generator"),

            ExprKind::Call(f, args) => {
                self.visit_expr(f);
                for arg in args {
                    self.visit_expr(arg);
                }

                self.handle_uninhabited_return(expr);
            }
            ExprKind::MethodCall(_, receiver, exprs, _) => {
                self.visit_expr(receiver);
                for expr in exprs {
                    self.visit_expr(expr);
                }

                self.handle_uninhabited_return(expr);
            }

            ExprKind::AddrOf(..)
            | ExprKind::Array(..)
            // FIXME(eholk): We probably need special handling for AssignOps. The ScopeTree builder
            // in region.rs runs both lhs then rhs and rhs then lhs and then sets all yields to be
            // the latest they show up in either traversal. With the older scope-based
            // approximation, this was fine, but it's probably not right now. What we probably want
            // to do instead is still run both orders, but consider anything that showed up as a
            // yield in either order.
            | ExprKind::AssignOp(..)
            | ExprKind::Binary(..)
            | ExprKind::Block(..)
            | ExprKind::Cast(..)
            | ExprKind::Closure { .. }
            | ExprKind::ConstBlock(..)
            | ExprKind::DropTemps(..)
            | ExprKind::Err(_)
            | ExprKind::Field(..)
            | ExprKind::Index(..)
            | ExprKind::InlineAsm(..)
            | ExprKind::OffsetOf(..)
            | ExprKind::Let(..)
            | ExprKind::Lit(..)
            | ExprKind::Path(..)
            | ExprKind::Repeat(..)
            | ExprKind::Ret(..)
            | ExprKind::Struct(..)
            | ExprKind::Tup(..)
            | ExprKind::Type(..)
            | ExprKind::Unary(..)
            | ExprKind::Yield(..) => intravisit::walk_expr(self, expr),
        }

        self.expr_index = self.expr_index + 1;
        self.drop_ranges.add_node_mapping(expr.hir_id, self.expr_index);
        self.consume_expr(expr);
        if let Some(expr) = reinit {
            self.reinit_expr(expr);
        }
    }

    fn visit_pat(&mut self, pat: &'tcx hir::Pat<'tcx>) {
        intravisit::walk_pat(self, pat);

        // Increment expr_count here to match what InteriorVisitor expects.
        self.expr_index = self.expr_index + 1;

        // Save a node mapping to get better CFG visualization
        self.drop_ranges.add_node_mapping(pat.hir_id, self.expr_index);
    }
}

impl DropRangesBuilder {
    fn new(
        tracked_values: impl Iterator<Item = TrackedValue>,
        hir: Map<'_>,
        num_exprs: usize,
    ) -> Self {
        let mut tracked_value_map = FxHashMap::<_, TrackedValueIndex>::default();
        let mut next = <_>::from(0u32);
        for value in tracked_values {
            for_each_consumable(hir, value, |value| {
                if let std::collections::hash_map::Entry::Vacant(e) = tracked_value_map.entry(value)
                {
                    e.insert(next);
                    next = next + 1;
                }
            });
        }
        debug!("hir_id_map: {:#?}", tracked_value_map);
        let num_values = tracked_value_map.len();
        Self {
            tracked_value_map,
            nodes: IndexVec::from_fn_n(|_| NodeInfo::new(num_values), num_exprs + 1),
            deferred_edges: <_>::default(),
            post_order_map: <_>::default(),
        }
    }

    fn tracked_value_index(&self, tracked_value: TrackedValue) -> TrackedValueIndex {
        *self.tracked_value_map.get(&tracked_value).unwrap()
    }

    /// Adds an entry in the mapping from HirIds to PostOrderIds
    ///
    /// Needed so that `add_control_edge_hir_id` can work.
    fn add_node_mapping(&mut self, node_hir_id: HirId, post_order_id: PostOrderId) {
        self.post_order_map.insert(node_hir_id, post_order_id);
    }

    /// Like add_control_edge, but uses a hir_id as the target.
    ///
    /// This can be used for branches where we do not know the PostOrderId of the target yet,
    /// such as when handling `break` or `continue`.
    fn add_control_edge_hir_id(&mut self, from: PostOrderId, to: HirId) {
        self.deferred_edges.push((from, to));
    }

    fn drop_at(&mut self, value: TrackedValue, location: PostOrderId) {
        let value = self.tracked_value_index(value);
        self.node_mut(location).drops.push(value);
    }

    fn reinit_at(&mut self, value: TrackedValue, location: PostOrderId) {
        let value = match self.tracked_value_map.get(&value) {
            Some(value) => *value,
            // If there's no value, this is never consumed and therefore is never dropped. We can
            // ignore this.
            None => return,
        };
        self.node_mut(location).reinits.push(value);
    }

    /// Looks up PostOrderId for any control edges added by HirId and adds a proper edge for them.
    ///
    /// Should be called after visiting the HIR but before solving the control flow, otherwise some
    /// edges will be missed.
    fn process_deferred_edges(&mut self) {
        trace!("processing deferred edges. post_order_map={:#?}", self.post_order_map);
        let mut edges = vec![];
        swap(&mut edges, &mut self.deferred_edges);
        edges.into_iter().for_each(|(from, to)| {
            trace!("Adding deferred edge from {:?} to {:?}", from, to);
            let to = *self.post_order_map.get(&to).expect("Expression ID not found");
            trace!("target edge PostOrderId={:?}", to);
            self.add_control_edge(from, to)
        });
    }
}
