use super::{for_each_consumable, record_consumed_borrow::ExprUseDelegate, DropRanges};
use hir::{
    intravisit::{self, NestedVisitorMap, Visitor},
    Expr, ExprKind, Guard, HirId, HirIdMap, HirIdSet,
};
use rustc_hir as hir;
use rustc_middle::hir::map::Map;

/// This struct is used to gather the information for `DropRanges` to determine the regions of the
/// HIR tree for which a value is dropped.
///
/// We are interested in points where a variables is dropped or initialized, and the control flow
/// of the code. We identify locations in code by their post-order traversal index, so it is
/// important for this traversal to match that in `RegionResolutionVisitor` and `InteriorVisitor`.
pub struct DropRangeVisitor<'tcx> {
    hir: Map<'tcx>,
    /// Maps a HirId to a set of HirIds that are dropped by that node.
    ///
    /// See also the more detailed comment on `ExprUseDelegate.consumed_places`.
    consumed_places: HirIdMap<HirIdSet>,
    borrowed_places: HirIdSet,
    drop_ranges: DropRanges,
    expr_count: usize,
}

impl<'tcx> DropRangeVisitor<'tcx> {
    pub fn from_uses(uses: ExprUseDelegate<'tcx>, num_exprs: usize) -> Self {
        debug!("consumed_places: {:?}", uses.consumed_places);
        let drop_ranges = DropRanges::new(
            uses.consumed_places.iter().flat_map(|(_, places)| places.iter().copied()),
            &uses.hir,
            num_exprs,
        );
        Self {
            hir: uses.hir,
            consumed_places: uses.consumed_places,
            borrowed_places: uses.borrowed_places,
            drop_ranges,
            expr_count: 0,
        }
    }

    pub fn into_drop_ranges(self) -> DropRanges {
        self.drop_ranges
    }

    fn record_drop(&mut self, hir_id: HirId) {
        if self.borrowed_places.contains(&hir_id) {
            debug!("not marking {:?} as dropped because it is borrowed at some point", hir_id);
        } else {
            debug!("marking {:?} as dropped at {}", hir_id, self.expr_count);
            let count = self.expr_count;
            self.drop_ranges.drop_at(hir_id, count);
        }
    }

    /// ExprUseVisitor's consume callback doesn't go deep enough for our purposes in all
    /// expressions. This method consumes a little deeper into the expression when needed.
    fn consume_expr(&mut self, expr: &hir::Expr<'_>) {
        debug!("consuming expr {:?}, count={}", expr.hir_id, self.expr_count);
        let places = self
            .consumed_places
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
            let location = self.expr_count;
            debug!("reinitializing {:?} at {}", hir_id, location);
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
            ExprKind::If(test, if_true, if_false) => {
                self.visit_expr(test);

                let fork = self.expr_count;

                self.drop_ranges.add_control_edge(fork, self.expr_count + 1);
                self.visit_expr(if_true);
                let true_end = self.expr_count;

                self.drop_ranges.add_control_edge(fork, self.expr_count + 1);
                if let Some(if_false) = if_false {
                    self.visit_expr(if_false);
                }

                self.drop_ranges.add_control_edge(true_end, self.expr_count + 1);
            }
            ExprKind::Assign(lhs, rhs, _) => {
                self.visit_expr(lhs);
                self.visit_expr(rhs);

                reinit = Some(lhs);
            }
            ExprKind::Loop(body, ..) => {
                let loop_begin = self.expr_count + 1;
                self.visit_block(body);
                self.drop_ranges.add_control_edge(self.expr_count, loop_begin);
            }
            ExprKind::Match(scrutinee, arms, ..) => {
                self.visit_expr(scrutinee);

                let fork = self.expr_count;
                let arm_end_ids = arms
                    .iter()
                    .map(|hir::Arm { pat, body, guard, .. }| {
                        self.drop_ranges.add_control_edge(fork, self.expr_count + 1);
                        self.visit_pat(pat);
                        match guard {
                            Some(Guard::If(expr)) => self.visit_expr(expr),
                            Some(Guard::IfLet(pat, expr)) => {
                                self.visit_pat(pat);
                                self.visit_expr(expr);
                            }
                            None => (),
                        }
                        self.visit_expr(body);
                        self.expr_count
                    })
                    .collect::<Vec<_>>();
                arm_end_ids.into_iter().for_each(|arm_end| {
                    self.drop_ranges.add_control_edge(arm_end, self.expr_count + 1)
                });
            }
            ExprKind::Break(hir::Destination { target_id: Ok(target), .. }, ..)
            | ExprKind::Continue(hir::Destination { target_id: Ok(target), .. }, ..) => {
                self.drop_ranges.add_control_edge_hir_id(self.expr_count, target);
            }

            _ => intravisit::walk_expr(self, expr),
        }

        self.expr_count += 1;
        self.drop_ranges.add_node_mapping(expr.hir_id, self.expr_count);
        self.consume_expr(expr);
        if let Some(expr) = reinit {
            self.reinit_expr(expr);
        }
    }

    fn visit_pat(&mut self, pat: &'tcx hir::Pat<'tcx>) {
        intravisit::walk_pat(self, pat);

        // Increment expr_count here to match what InteriorVisitor expects.
        self.expr_count += 1;
    }
}
