//! Drop range analysis finds the portions of the tree where a value is guaranteed to be dropped
//! (i.e. moved, uninitialized, etc.). This is used to exclude the types of those values from the
//! generator type. See `InteriorVisitor::record` for where the results of this analysis are used.
//!
//! There are three phases to this analysis:
//! 1. Use `ExprUseVisitor` to identify the interesting values that are consumed and borrowed.
//! 2. Use `DropRangeVisitor` to find where the interesting values are dropped or reinitialized,
//!    and also build a control flow graph.
//! 3. Use `DropRanges::propagate_to_fixpoint` to flow the dropped/reinitialized information through
//!    the CFG and find the exact points where we know a value is definitely dropped.
//!
//! The end result is a data structure that maps the post-order index of each node in the HIR tree
//! to a set of values that are known to be dropped at that location.

use std::collections::BTreeMap;
use std::fmt::Debug;
use std::mem::swap;

use hir::intravisit::{self, NestedVisitorMap, Visitor};
use hir::{Expr, ExprKind, Guard, HirId, HirIdMap, HirIdSet, Node};
use rustc_graphviz as dot;
use rustc_hir as hir;
use rustc_index::bit_set::BitSet;
use rustc_index::vec::IndexVec;
use rustc_middle::hir::map::Map;
use rustc_middle::hir::place::{Place, PlaceBase};
use rustc_middle::ty;

use crate::expr_use_visitor;

/// Works with ExprUseVisitor to find interesting values for the drop range analysis.
///
/// Interesting values are those that are either dropped or borrowed. For dropped values, we also
/// record the parent expression, which is the point where the drop actually takes place.
pub struct ExprUseDelegate<'tcx> {
    hir: Map<'tcx>,
    /// Maps a HirId to a set of HirIds that are dropped by that node.
    consumed_places: HirIdMap<HirIdSet>,
    borrowed_places: HirIdSet,
}

impl<'tcx> ExprUseDelegate<'tcx> {
    pub fn new(hir: Map<'tcx>) -> Self {
        Self { hir, consumed_places: <_>::default(), borrowed_places: <_>::default() }
    }

    fn mark_consumed(&mut self, consumer: HirId, target: HirId) {
        if !self.consumed_places.contains_key(&consumer) {
            self.consumed_places.insert(consumer, <_>::default());
        }
        self.consumed_places.get_mut(&consumer).map(|places| places.insert(target));
    }
}

impl<'tcx> expr_use_visitor::Delegate<'tcx> for ExprUseDelegate<'tcx> {
    fn consume(
        &mut self,
        place_with_id: &expr_use_visitor::PlaceWithHirId<'tcx>,
        diag_expr_id: hir::HirId,
    ) {
        let parent = match self.hir.find_parent_node(place_with_id.hir_id) {
            Some(parent) => parent,
            None => place_with_id.hir_id,
        };
        debug!(
            "consume {:?}; diag_expr_id={:?}, using parent {:?}",
            place_with_id, diag_expr_id, parent
        );
        self.mark_consumed(parent, place_with_id.hir_id);
        place_hir_id(&place_with_id.place).map(|place| self.mark_consumed(parent, place));
    }

    fn borrow(
        &mut self,
        place_with_id: &expr_use_visitor::PlaceWithHirId<'tcx>,
        _diag_expr_id: hir::HirId,
        _bk: rustc_middle::ty::BorrowKind,
    ) {
        place_hir_id(&place_with_id.place).map(|place| self.borrowed_places.insert(place));
    }

    fn mutate(
        &mut self,
        _assignee_place: &expr_use_visitor::PlaceWithHirId<'tcx>,
        _diag_expr_id: hir::HirId,
    ) {
    }

    fn fake_read(
        &mut self,
        _place: expr_use_visitor::Place<'tcx>,
        _cause: rustc_middle::mir::FakeReadCause,
        _diag_expr_id: hir::HirId,
    ) {
    }
}

/// Gives the hir_id associated with a place if one exists. This is the hir_id that we want to
/// track for a value in the drop range analysis.
fn place_hir_id(place: &Place<'_>) -> Option<HirId> {
    match place.base {
        PlaceBase::Rvalue | PlaceBase::StaticItem => None,
        PlaceBase::Local(hir_id)
        | PlaceBase::Upvar(ty::UpvarId { var_path: ty::UpvarPath { hir_id }, .. }) => Some(hir_id),
    }
}

/// This struct is used to gather the information for `DropRanges` to determine the regions of the
/// HIR tree for which a value is dropped.
///
/// We are interested in points where a variables is dropped or initialized, and the control flow
/// of the code. We identify locations in code by their post-order traversal index, so it is
/// important for this traversal to match that in `RegionResolutionVisitor` and `InteriorVisitor`.
pub struct DropRangeVisitor<'tcx> {
    hir: Map<'tcx>,
    /// Maps a HirId to a set of HirIds that are dropped by that node.
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

/// Applies `f` to consumable portion of a HIR node.
///
/// The `node` parameter should be the result of calling `Map::find(place)`.
fn for_each_consumable(place: HirId, node: Option<Node<'_>>, mut f: impl FnMut(HirId)) {
    f(place);
    if let Some(Node::Expr(expr)) = node {
        match expr.kind {
            hir::ExprKind::Path(hir::QPath::Resolved(
                _,
                hir::Path { res: hir::def::Res::Local(hir_id), .. },
            )) => {
                f(*hir_id);
            }
            _ => (),
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

rustc_index::newtype_index! {
    pub struct PostOrderId {
        DEBUG_FORMAT = "id({})",
    }
}

rustc_index::newtype_index! {
    pub struct HirIdIndex {
        DEBUG_FORMAT = "hidx({})",
    }
}

pub struct DropRanges {
    hir_id_map: HirIdMap<HirIdIndex>,
    nodes: IndexVec<PostOrderId, NodeInfo>,
    deferred_edges: Vec<(usize, HirId)>,
    // FIXME: This should only be used for loops and break/continue.
    post_order_map: HirIdMap<usize>,
}

impl Debug for DropRanges {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DropRanges")
            .field("hir_id_map", &self.hir_id_map)
            .field("post_order_maps", &self.post_order_map)
            .field("nodes", &self.nodes.iter_enumerated().collect::<BTreeMap<_, _>>())
            .finish()
    }
}

/// DropRanges keeps track of what values are definitely dropped at each point in the code.
///
/// Values of interest are defined by the hir_id of their place. Locations in code are identified
/// by their index in the post-order traversal. At its core, DropRanges maps
/// (hir_id, post_order_id) -> bool, where a true value indicates that the value is definitely
/// dropped at the point of the node identified by post_order_id.
impl DropRanges {
    pub fn new(hir_ids: impl Iterator<Item = HirId>, hir: &Map<'_>, num_exprs: usize) -> Self {
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

    pub fn is_dropped_at(&mut self, hir_id: HirId, location: usize) -> bool {
        self.hir_id_map
            .get(&hir_id)
            .copied()
            .map_or(false, |hir_id| self.expect_node(location.into()).drop_state.contains(hir_id))
    }

    /// Returns the number of values (hir_ids) that are tracked
    fn num_values(&self) -> usize {
        self.hir_id_map.len()
    }

    /// Adds an entry in the mapping from HirIds to PostOrderIds
    ///
    /// Needed so that `add_control_edge_hir_id` can work.
    pub fn add_node_mapping(&mut self, hir_id: HirId, post_order_id: usize) {
        self.post_order_map.insert(hir_id, post_order_id);
    }

    /// Returns a reference to the NodeInfo for a node, panicking if it does not exist
    fn expect_node(&self, id: PostOrderId) -> &NodeInfo {
        &self.nodes[id]
    }

    fn node_mut(&mut self, id: PostOrderId) -> &mut NodeInfo {
        let size = self.num_values();
        self.nodes.ensure_contains_elem(id, || NodeInfo::new(size));
        &mut self.nodes[id]
    }

    pub fn add_control_edge(&mut self, from: usize, to: usize) {
        trace!("adding control edge from {} to {}", from, to);
        self.node_mut(from.into()).successors.push(to.into());
    }

    /// Like add_control_edge, but uses a hir_id as the target.
    ///
    /// This can be used for branches where we do not know the PostOrderId of the target yet,
    /// such as when handling `break` or `continue`.
    pub fn add_control_edge_hir_id(&mut self, from: usize, to: HirId) {
        self.deferred_edges.push((from, to));
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
            trace!("Adding deferred edge from {} to {}", from, to);
            self.add_control_edge(from, to)
        });
    }

    pub fn drop_at(&mut self, value: HirId, location: usize) {
        let value = self.hidx(value);
        self.node_mut(location.into()).drops.push(value);
    }

    pub fn reinit_at(&mut self, value: HirId, location: usize) {
        let value = match self.hir_id_map.get(&value) {
            Some(value) => *value,
            // If there's no value, this is never consumed and therefore is never dropped. We can
            // ignore this.
            None => return,
        };
        self.node_mut(location.into()).reinits.push(value);
    }

    pub fn propagate_to_fixpoint(&mut self) {
        trace!("before fixpoint: {:#?}", self);
        self.process_deferred_edges();
        let preds = self.compute_predecessors();

        trace!("predecessors: {:#?}", preds.iter_enumerated().collect::<BTreeMap<_, _>>());

        let mut propagate = || {
            let mut changed = false;
            for id in self.nodes.indices() {
                let old_state = self.nodes[id].drop_state.clone();
                if preds[id].len() != 0 {
                    self.nodes[id].drop_state = self.nodes[preds[id][0]].drop_state.clone();
                    for pred in &preds[id][1..] {
                        let state = self.nodes[*pred].drop_state.clone();
                        self.nodes[id].drop_state.intersect(&state);
                    }
                } else {
                    self.nodes[id].drop_state = if id.index() == 0 {
                        BitSet::new_empty(self.num_values())
                    } else {
                        // If we are not the start node and we have no predecessors, treat
                        // everything as dropped because there's no way to get here anyway.
                        BitSet::new_filled(self.num_values())
                    };
                };
                for drop in &self.nodes[id].drops.clone() {
                    self.nodes[id].drop_state.insert(*drop);
                }
                for reinit in &self.nodes[id].reinits.clone() {
                    self.nodes[id].drop_state.remove(*reinit);
                }

                changed |= old_state != self.nodes[id].drop_state;
            }

            changed
        };

        while propagate() {
            trace!("drop_state changed, re-running propagation");
        }

        trace!("after fixpoint: {:#?}", self);
    }

    fn compute_predecessors(&self) -> IndexVec<PostOrderId, Vec<PostOrderId>> {
        let mut preds = IndexVec::from_fn_n(|_| vec![], self.nodes.len());
        for (id, node) in self.nodes.iter_enumerated() {
            if node.successors.len() == 0 && id.index() != self.nodes.len() - 1 {
                preds[<_>::from(id.index() + 1)].push(id);
            } else {
                for succ in &node.successors {
                    preds[*succ].push(id);
                }
            }
        }
        preds
    }
}

impl<'a> dot::GraphWalk<'a> for DropRanges {
    type Node = PostOrderId;

    type Edge = (PostOrderId, PostOrderId);

    fn nodes(&'a self) -> dot::Nodes<'a, Self::Node> {
        self.nodes.iter_enumerated().map(|(i, _)| i).collect()
    }

    fn edges(&'a self) -> dot::Edges<'a, Self::Edge> {
        self.nodes
            .iter_enumerated()
            .flat_map(|(i, node)| {
                if node.successors.len() == 0 {
                    vec![(i, PostOrderId::from_usize(i.index() + 1))]
                } else {
                    node.successors.iter().map(move |&s| (i, s)).collect()
                }
            })
            .collect()
    }

    fn source(&'a self, edge: &Self::Edge) -> Self::Node {
        edge.0
    }

    fn target(&'a self, edge: &Self::Edge) -> Self::Node {
        edge.1
    }
}

impl<'a> dot::Labeller<'a> for DropRanges {
    type Node = PostOrderId;

    type Edge = (PostOrderId, PostOrderId);

    fn graph_id(&'a self) -> dot::Id<'a> {
        dot::Id::new("drop_ranges").unwrap()
    }

    fn node_id(&'a self, n: &Self::Node) -> dot::Id<'a> {
        dot::Id::new(format!("id{}", n.index())).unwrap()
    }

    fn node_label(&'a self, n: &Self::Node) -> dot::LabelText<'a> {
        dot::LabelText::LabelStr(
            format!(
                "{:?}, local_id: {}",
                n,
                self.post_order_map
                    .iter()
                    .find(|(_hir_id, &post_order_id)| post_order_id == n.index())
                    .map_or("<unknown>".into(), |(hir_id, _)| format!(
                        "{}",
                        hir_id.local_id.index()
                    ))
            )
            .into(),
        )
    }
}

#[derive(Debug)]
struct NodeInfo {
    /// IDs of nodes that can follow this one in the control flow
    ///
    /// If the vec is empty, then control proceeds to the next node.
    successors: Vec<PostOrderId>,

    /// List of hir_ids that are dropped by this node.
    drops: Vec<HirIdIndex>,

    /// List of hir_ids that are reinitialized by this node.
    reinits: Vec<HirIdIndex>,

    /// Set of values that are definitely dropped at this point.
    drop_state: BitSet<HirIdIndex>,
}

impl NodeInfo {
    fn new(num_values: usize) -> Self {
        Self {
            successors: vec![],
            drops: vec![],
            reinits: vec![],
            drop_state: BitSet::new_filled(num_values),
        }
    }
}
