use super::{
    for_each_consumable, record_consumed_borrow::ConsumedAndBorrowedPlaces, DropRangesBuilder,
    NodeInfo, PostOrderId, TrackedValue, TrackedValueIndex,
};
use hir::{
    intravisit::{self, Visitor},
    Body, Expr, ExprKind, Guard, HirId,
};
use rustc_data_structures::fx::FxHashMap;
use rustc_hir as hir;
use rustc_index::vec::IndexVec;
use rustc_middle::{
    hir::map::Map,
    ty::{TyCtxt, TypeckResults},
};
use std::mem::swap;

/// Traverses the body to find the control flow graph and locations for the
/// relevant places are dropped or reinitialized.
///
/// The resulting structure still needs to be iterated to a fixed point, which
/// can be done with propagate_to_fixpoint in cfg_propagate.
pub(super) fn build_control_flow_graph<'tcx>(
    hir: Map<'tcx>,
    tcx: TyCtxt<'tcx>,
    typeck_results: &TypeckResults<'tcx>,
    consumed_borrowed_places: ConsumedAndBorrowedPlaces,
    body: &'tcx Body<'tcx>,
    num_exprs: usize,
) -> DropRangesBuilder {
    let mut drop_range_visitor =
        DropRangeVisitor::new(hir, tcx, typeck_results, consumed_borrowed_places, num_exprs);
    intravisit::walk_body(&mut drop_range_visitor, body);

    drop_range_visitor.drop_ranges.process_deferred_edges();

    drop_range_visitor.drop_ranges
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
/// 3. Reinitializing through a field (e.g. `a.b.c = 5`) counds as a reinitialization of all of
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
/// ```rust
/// let mut a = (vec![0], vec![0]);
/// drop(a);
/// a.1 = vec![1];
/// // all of `a` is considered initialized
/// ```

struct DropRangeVisitor<'a, 'tcx> {
    hir: Map<'tcx>,
    places: ConsumedAndBorrowedPlaces,
    drop_ranges: DropRangesBuilder,
    expr_index: PostOrderId,
    tcx: TyCtxt<'tcx>,
    typeck_results: &'a TypeckResults<'tcx>,
}

impl<'a, 'tcx> DropRangeVisitor<'a, 'tcx> {
    fn new(
        hir: Map<'tcx>,
        tcx: TyCtxt<'tcx>,
        typeck_results: &'a TypeckResults<'tcx>,
        places: ConsumedAndBorrowedPlaces,
        num_exprs: usize,
    ) -> Self {
        debug!("consumed_places: {:?}", places.consumed);
        let drop_ranges = DropRangesBuilder::new(
            places.consumed.iter().flat_map(|(_, places)| places.iter().cloned()),
            hir,
            num_exprs,
        );
        Self { hir, places, drop_ranges, expr_index: PostOrderId::from_u32(0), typeck_results, tcx }
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
        debug!("consuming expr {:?}, count={:?}", expr.hir_id, self.expr_index);
        let places = self
            .places
            .consumed
            .get(&expr.hir_id)
            .map_or(vec![], |places| places.iter().cloned().collect());
        for place in places {
            for_each_consumable(self.hir, place, |value| self.record_drop(value));
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
            | ExprKind::Box(..)
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
            | ExprKind::Closure(..)
            | ExprKind::Block(..)
            | ExprKind::Assign(..)
            | ExprKind::AssignOp(..)
            | ExprKind::Index(..)
            | ExprKind::AddrOf(..)
            | ExprKind::Break(..)
            | ExprKind::Continue(..)
            | ExprKind::Ret(..)
            | ExprKind::InlineAsm(..)
            | ExprKind::Struct(..)
            | ExprKind::Repeat(..)
            | ExprKind::Yield(..)
            | ExprKind::Err => (),
        }
    }

    /// For an expression with an uninhabited return type (e.g. a function that returns !),
    /// this adds a self edge to to the CFG to model the fact that the function does not
    /// return.
    fn handle_uninhabited_return(&mut self, expr: &Expr<'tcx>) {
        let ty = self.typeck_results.expr_ty(expr);
        let ty = self.tcx.erase_regions(ty);
        let m = self.tcx.parent_module(expr.hir_id).to_def_id();
        let param_env = self.tcx.param_env(m.expect_local());
        if self.tcx.is_ty_uninhabited_from(m, ty, param_env) {
            // This function will not return. We model this fact as an infinite loop.
            self.drop_ranges.add_control_edge(self.expr_index + 1, self.expr_index + 1);
        }
    }
}

impl<'a, 'tcx> Visitor<'tcx> for DropRangeVisitor<'a, 'tcx> {
    fn visit_expr(&mut self, expr: &'tcx Expr<'tcx>) {
        let mut reinit = None;
        match expr.kind {
            ExprKind::Assign(lhs, rhs, _) => {
                self.visit_expr(lhs);
                self.visit_expr(rhs);

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
                            Some(Guard::IfLet(pat, expr)) => {
                                self.visit_pat(pat);
                                self.visit_expr(expr);
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

            ExprKind::Loop(body, ..) => {
                let loop_begin = self.expr_index + 1;
                if body.stmts.is_empty() && body.expr.is_none() {
                    // For empty loops we won't have updated self.expr_index after visiting the
                    // body, meaning we'd get an edge from expr_index to expr_index + 1, but
                    // instead we want an edge from expr_index + 1 to expr_index + 1.
                    self.drop_ranges.add_control_edge(loop_begin, loop_begin);
                } else {
                    self.visit_block(body);
                    self.drop_ranges.add_control_edge(self.expr_index, loop_begin);
                }
            }
            ExprKind::Break(hir::Destination { target_id: Ok(target), .. }, ..)
            | ExprKind::Continue(hir::Destination { target_id: Ok(target), .. }, ..) => {
                self.drop_ranges.add_control_edge_hir_id(self.expr_index, target);
            }

            ExprKind::Call(f, args) => {
                self.visit_expr(f);
                for arg in args {
                    self.visit_expr(arg);
                }

                self.handle_uninhabited_return(expr);
            }
            ExprKind::MethodCall(_, exprs, _) => {
                for expr in exprs {
                    self.visit_expr(expr);
                }

                self.handle_uninhabited_return(expr);
            }

            ExprKind::AddrOf(..)
            | ExprKind::Array(..)
            | ExprKind::AssignOp(..)
            | ExprKind::Binary(..)
            | ExprKind::Block(..)
            | ExprKind::Box(..)
            | ExprKind::Break(..)
            | ExprKind::Cast(..)
            | ExprKind::Closure(..)
            | ExprKind::ConstBlock(..)
            | ExprKind::Continue(..)
            | ExprKind::DropTemps(..)
            | ExprKind::Err
            | ExprKind::Field(..)
            | ExprKind::Index(..)
            | ExprKind::InlineAsm(..)
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
                if !tracked_value_map.contains_key(&value) {
                    tracked_value_map.insert(value, next);
                    next = next + 1;
                }
            });
        }
        debug!("hir_id_map: {:?}", tracked_value_map);
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
        self.node_mut(location.into()).drops.push(value);
    }

    fn reinit_at(&mut self, value: TrackedValue, location: PostOrderId) {
        let value = match self.tracked_value_map.get(&value) {
            Some(value) => *value,
            // If there's no value, this is never consumed and therefore is never dropped. We can
            // ignore this.
            None => return,
        };
        self.node_mut(location.into()).reinits.push(value);
    }

    /// Looks up PostOrderId for any control edges added by HirId and adds a proper edge for them.
    ///
    /// Should be called after visiting the HIR but before solving the control flow, otherwise some
    /// edges will be missed.
    fn process_deferred_edges(&mut self) {
        let mut edges = vec![];
        swap(&mut edges, &mut self.deferred_edges);
        edges.into_iter().for_each(|(from, to)| {
            let to = *self.post_order_map.get(&to).expect("Expression ID not found");
            trace!("Adding deferred edge from {:?} to {:?}", from, to);
            self.add_control_edge(from, to)
        });
    }
}
