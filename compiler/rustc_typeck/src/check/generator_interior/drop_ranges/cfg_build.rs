use super::{
    for_each_consumable, record_consumed_borrow::ConsumedAndBorrowedPlaces, DropRangesBuilder,
    HirIdIndex, NodeInfo, PostOrderId,
};
use hir::{
    intravisit::{self, NestedVisitorMap, Visitor},
    Body, Expr, ExprKind, Guard, HirId, HirIdMap,
};
use rustc_hir as hir;
use rustc_index::vec::IndexVec;
use rustc_middle::hir::map::Map;
use std::mem::swap;

/// Traverses the body to find the control flow graph and locations for the
/// relevant places are dropped or reinitialized.
///
/// The resulting structure still needs to be iterated to a fixed point, which
/// can be done with propagate_to_fixpoint in cfg_propagate.
pub(super) fn build_control_flow_graph<'tcx>(
    hir: Map<'tcx>,
    consumed_borrowed_places: ConsumedAndBorrowedPlaces,
    body: &'tcx Body<'tcx>,
    num_exprs: usize,
) -> DropRangesBuilder {
    let mut drop_range_visitor = DropRangeVisitor::new(hir, consumed_borrowed_places, num_exprs);
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
struct DropRangeVisitor<'tcx> {
    hir: Map<'tcx>,
    places: ConsumedAndBorrowedPlaces,
    drop_ranges: DropRangesBuilder,
    expr_index: PostOrderId,
}

impl<'tcx> DropRangeVisitor<'tcx> {
    fn new(hir: Map<'tcx>, places: ConsumedAndBorrowedPlaces, num_exprs: usize) -> Self {
        debug!("consumed_places: {:?}", places.consumed);
        let drop_ranges = DropRangesBuilder::new(
            places.consumed.iter().flat_map(|(_, places)| places.iter().copied()),
            hir,
            num_exprs,
        );
        Self { hir, places, drop_ranges, expr_index: PostOrderId::from_u32(0) }
    }

    fn record_drop(&mut self, hir_id: HirId) {
        if self.places.borrowed.contains(&hir_id) {
            debug!("not marking {:?} as dropped because it is borrowed at some point", hir_id);
        } else {
            debug!("marking {:?} as dropped at {:?}", hir_id, self.expr_index);
            let count = self.expr_index;
            self.drop_ranges.drop_at(hir_id, count);
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
            for_each_consumable(place, self.hir.find(place), |hir_id| self.record_drop(hir_id));
        }
    }

    fn reinit_expr(&mut self, expr: &hir::Expr<'_>) {
        if let ExprKind::Path(hir::QPath::Resolved(
            _,
            hir::Path { res: hir::def::Res::Local(hir_id), .. },
        )) = expr.kind
        {
            let location = self.expr_index;
            debug!("reinitializing {:?} at {:?}", hir_id, location);
            self.drop_ranges.reinit_at(*hir_id, location);
        } else {
            debug!("reinitializing {:?} is not supported", expr);
        }
    }
}

impl<'tcx> Visitor<'tcx> for DropRangeVisitor<'tcx> {
    type Map = intravisit::ErasedMap<'tcx>;

    fn nested_visit_map(&mut self) -> NestedVisitorMap<Self::Map> {
        NestedVisitorMap::None
    }

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
                self.visit_expr(scrutinee);

                let (guard_exit, arm_end_ids) = arms.iter().fold(
                    (self.expr_index, vec![]),
                    |(incoming_edge, mut arm_end_ids), hir::Arm { pat, body, guard, .. }| {
                        self.drop_ranges.add_control_edge(incoming_edge, self.expr_index + 1);
                        self.visit_pat(pat);
                        match guard {
                            Some(Guard::If(expr)) => self.visit_expr(expr),
                            Some(Guard::IfLet(pat, expr)) => {
                                self.visit_pat(pat);
                                self.visit_expr(expr);
                            }
                            None => (),
                        }
                        let to_next_arm = self.expr_index;
                        // The default edge does not get added since we also have an explicit edge,
                        // so we also need to add an edge to the next node as well.
                        self.drop_ranges.add_control_edge(self.expr_index, self.expr_index + 1);
                        self.visit_expr(body);
                        arm_end_ids.push(self.expr_index);
                        (to_next_arm, arm_end_ids)
                    },
                );
                self.drop_ranges.add_control_edge(guard_exit, self.expr_index + 1);
                arm_end_ids.into_iter().for_each(|arm_end| {
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

            ExprKind::AddrOf(..)
            | ExprKind::Array(..)
            | ExprKind::AssignOp(..)
            | ExprKind::Binary(..)
            | ExprKind::Block(..)
            | ExprKind::Box(..)
            | ExprKind::Break(..)
            | ExprKind::Call(..)
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
            | ExprKind::LlvmInlineAsm(..)
            | ExprKind::MethodCall(..)
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
    fn new(hir_ids: impl Iterator<Item = HirId>, hir: Map<'_>, num_exprs: usize) -> Self {
        let mut hir_id_map = HirIdMap::<HirIdIndex>::default();
        let mut next = <_>::from(0u32);
        for hir_id in hir_ids {
            for_each_consumable(hir_id, hir.find(hir_id), |hir_id| {
                if !hir_id_map.contains_key(&hir_id) {
                    hir_id_map.insert(hir_id, next);
                    next = <_>::from(next.index() + 1);
                }
            });
        }
        debug!("hir_id_map: {:?}", hir_id_map);
        let num_values = hir_id_map.len();
        Self {
            hir_id_map,
            nodes: IndexVec::from_fn_n(|_| NodeInfo::new(num_values), num_exprs + 1),
            deferred_edges: <_>::default(),
            post_order_map: <_>::default(),
        }
    }

    fn hidx(&self, hir_id: HirId) -> HirIdIndex {
        *self.hir_id_map.get(&hir_id).unwrap()
    }

    /// Adds an entry in the mapping from HirIds to PostOrderIds
    ///
    /// Needed so that `add_control_edge_hir_id` can work.
    fn add_node_mapping(&mut self, hir_id: HirId, post_order_id: PostOrderId) {
        self.post_order_map.insert(hir_id, post_order_id);
    }

    /// Like add_control_edge, but uses a hir_id as the target.
    ///
    /// This can be used for branches where we do not know the PostOrderId of the target yet,
    /// such as when handling `break` or `continue`.
    fn add_control_edge_hir_id(&mut self, from: PostOrderId, to: HirId) {
        self.deferred_edges.push((from, to));
    }

    fn drop_at(&mut self, value: HirId, location: PostOrderId) {
        let value = self.hidx(value);
        self.node_mut(location.into()).drops.push(value);
    }

    fn reinit_at(&mut self, value: HirId, location: PostOrderId) {
        let value = match self.hir_id_map.get(&value) {
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
