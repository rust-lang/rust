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

use self::cfg_build::build_control_flow_graph;
use self::record_consumed_borrow::find_consumed_and_borrowed;
use crate::check::FnCtxt;
use hir::def_id::DefId;
use hir::{Body, HirId, HirIdMap, Node};
use rustc_data_structures::fx::FxHashMap;
use rustc_hir as hir;
use rustc_index::bit_set::BitSet;
use rustc_index::vec::IndexVec;
use rustc_middle::hir::map::Map;
use rustc_middle::hir::place::{PlaceBase, PlaceWithHirId};
use rustc_middle::ty;
use std::collections::BTreeMap;
use std::fmt::Debug;

mod cfg_build;
mod cfg_propagate;
mod cfg_visualize;
mod record_consumed_borrow;

pub fn compute_drop_ranges<'a, 'tcx>(
    fcx: &'a FnCtxt<'a, 'tcx>,
    def_id: DefId,
    body: &'tcx Body<'tcx>,
) -> DropRanges {
    if fcx.sess().opts.debugging_opts.drop_tracking {
        let consumed_borrowed_places = find_consumed_and_borrowed(fcx, def_id, body);

        let num_exprs = fcx.tcx.region_scope_tree(def_id).body_expr_count(body.id()).unwrap_or(0);
        let mut drop_ranges = build_control_flow_graph(
            fcx.tcx.hir(),
            fcx.tcx,
            &fcx.typeck_results.borrow(),
            consumed_borrowed_places,
            body,
            num_exprs,
        );

        drop_ranges.propagate_to_fixpoint();

        DropRanges { tracked_value_map: drop_ranges.tracked_value_map, nodes: drop_ranges.nodes }
    } else {
        // If drop range tracking is not enabled, skip all the analysis and produce an
        // empty set of DropRanges.
        DropRanges { tracked_value_map: FxHashMap::default(), nodes: IndexVec::new() }
    }
}

/// Applies `f` to consumable node in the HIR subtree pointed to by `place`.
///
/// This includes the place itself, and if the place is a reference to a local
/// variable then `f` is also called on the HIR node for that variable as well.
///
/// For example, if `place` points to `foo()`, then `f` is called once for the
/// result of `foo`. On the other hand, if `place` points to `x` then `f` will
/// be called both on the `ExprKind::Path` node that represents the expression
/// as well as the HirId of the local `x` itself.
fn for_each_consumable<'tcx>(hir: Map<'tcx>, place: TrackedValue, mut f: impl FnMut(TrackedValue)) {
    f(place);
    let node = hir.find(place.hir_id());
    if let Some(Node::Expr(expr)) = node {
        match expr.kind {
            hir::ExprKind::Path(hir::QPath::Resolved(
                _,
                hir::Path { res: hir::def::Res::Local(hir_id), .. },
            )) => {
                f(TrackedValue::Variable(*hir_id));
            }
            _ => (),
        }
    }
}

rustc_index::newtype_index! {
    pub struct PostOrderId {
        DEBUG_FORMAT = "id({})",
    }
}

rustc_index::newtype_index! {
    pub struct TrackedValueIndex {
        DEBUG_FORMAT = "hidx({})",
    }
}

/// Identifies a value whose drop state we need to track.
#[derive(PartialEq, Eq, Hash, Debug, Clone, Copy)]
enum TrackedValue {
    /// Represents a named variable, such as a let binding, parameter, or upvar.
    ///
    /// The HirId points to the variable's definition site.
    Variable(HirId),
    /// A value produced as a result of an expression.
    ///
    /// The HirId points to the expression that returns this value.
    Temporary(HirId),
}

impl TrackedValue {
    fn hir_id(&self) -> HirId {
        match self {
            TrackedValue::Variable(hir_id) | TrackedValue::Temporary(hir_id) => *hir_id,
        }
    }

    fn from_place_with_projections_allowed(place_with_id: &PlaceWithHirId<'_>) -> Self {
        match place_with_id.place.base {
            PlaceBase::Rvalue | PlaceBase::StaticItem => {
                TrackedValue::Temporary(place_with_id.hir_id)
            }
            PlaceBase::Local(hir_id)
            | PlaceBase::Upvar(ty::UpvarId { var_path: ty::UpvarPath { hir_id }, .. }) => {
                TrackedValue::Variable(hir_id)
            }
        }
    }
}

/// Represents a reason why we might not be able to convert a HirId or Place
/// into a tracked value.
#[derive(Debug)]
enum TrackedValueConversionError {
    /// Place projects are not currently supported.
    ///
    /// The reasoning around these is kind of subtle, so we choose to be more
    /// conservative around these for now. There is not reason in theory we
    /// cannot support these, we just have not implemented it yet.
    PlaceProjectionsNotSupported,
}

impl TryFrom<&PlaceWithHirId<'_>> for TrackedValue {
    type Error = TrackedValueConversionError;

    fn try_from(place_with_id: &PlaceWithHirId<'_>) -> Result<Self, Self::Error> {
        if !place_with_id.place.projections.is_empty() {
            debug!(
                "TrackedValue from PlaceWithHirId: {:?} has projections, which are not supported.",
                place_with_id
            );
            return Err(TrackedValueConversionError::PlaceProjectionsNotSupported);
        }

        Ok(TrackedValue::from_place_with_projections_allowed(place_with_id))
    }
}

pub struct DropRanges {
    tracked_value_map: FxHashMap<TrackedValue, TrackedValueIndex>,
    nodes: IndexVec<PostOrderId, NodeInfo>,
}

impl DropRanges {
    pub fn is_dropped_at(&self, hir_id: HirId, location: usize) -> bool {
        self.tracked_value_map
            .get(&TrackedValue::Temporary(hir_id))
            .or(self.tracked_value_map.get(&TrackedValue::Variable(hir_id)))
            .cloned()
            .map_or(false, |tracked_value_id| {
                self.expect_node(location.into()).drop_state.contains(tracked_value_id)
            })
    }

    /// Returns a reference to the NodeInfo for a node, panicking if it does not exist
    fn expect_node(&self, id: PostOrderId) -> &NodeInfo {
        &self.nodes[id]
    }
}

/// Tracks information needed to compute drop ranges.
struct DropRangesBuilder {
    /// The core of DropRangesBuilder is a set of nodes, which each represent
    /// one expression. We primarily refer to them by their index in a
    /// post-order traversal of the HIR tree,  since this is what
    /// generator_interior uses to talk about yield positions.
    ///
    /// This IndexVec keeps the relevant details for each node. See the
    /// NodeInfo struct for more details, but this information includes things
    /// such as the set of control-flow successors, which variables are dropped
    /// or reinitialized, and whether each variable has been inferred to be
    /// known-dropped or potentially reintiialized at each point.
    nodes: IndexVec<PostOrderId, NodeInfo>,
    /// We refer to values whose drop state we are tracking by the HirId of
    /// where they are defined. Within a NodeInfo, however, we store the
    /// drop-state in a bit vector indexed by a HirIdIndex
    /// (see NodeInfo::drop_state). The hir_id_map field stores the mapping
    /// from HirIds to the HirIdIndex that is used to represent that value in
    /// bitvector.
    tracked_value_map: FxHashMap<TrackedValue, TrackedValueIndex>,

    /// When building the control flow graph, we don't always know the
    /// post-order index of the target node at the point we encounter it.
    /// For example, this happens with break and continue. In those cases,
    /// we store a pair of the PostOrderId of the source and the HirId
    /// of the target. Once we have gathered all of these edges, we make a
    /// pass over the set of deferred edges (see process_deferred_edges in
    /// cfg_build.rs), look up the PostOrderId for the target (since now the
    /// post-order index for all nodes is known), and add missing control flow
    /// edges.
    deferred_edges: Vec<(PostOrderId, HirId)>,
    /// This maps HirIds of expressions to their post-order index. It is
    /// used in process_deferred_edges to correctly add back-edges.
    post_order_map: HirIdMap<PostOrderId>,
}

impl Debug for DropRangesBuilder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DropRanges")
            .field("hir_id_map", &self.tracked_value_map)
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
impl DropRangesBuilder {
    /// Returns the number of values (hir_ids) that are tracked
    fn num_values(&self) -> usize {
        self.tracked_value_map.len()
    }

    fn node_mut(&mut self, id: PostOrderId) -> &mut NodeInfo {
        let size = self.num_values();
        self.nodes.ensure_contains_elem(id, || NodeInfo::new(size));
        &mut self.nodes[id]
    }

    fn add_control_edge(&mut self, from: PostOrderId, to: PostOrderId) {
        trace!("adding control edge from {:?} to {:?}", from, to);
        self.node_mut(from).successors.push(to);
    }
}

#[derive(Debug)]
struct NodeInfo {
    /// IDs of nodes that can follow this one in the control flow
    ///
    /// If the vec is empty, then control proceeds to the next node.
    successors: Vec<PostOrderId>,

    /// List of hir_ids that are dropped by this node.
    drops: Vec<TrackedValueIndex>,

    /// List of hir_ids that are reinitialized by this node.
    reinits: Vec<TrackedValueIndex>,

    /// Set of values that are definitely dropped at this point.
    drop_state: BitSet<TrackedValueIndex>,
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
