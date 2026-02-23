use rustc_data_structures::fx::FxIndexSet;
use rustc_data_structures::graph::scc::Sccs;
use rustc_data_structures::graph::{DirectedGraph, Successors};
use rustc_data_structures::unord::UnordMap;
use rustc_hir::def_id::DefId;
use rustc_index::{Idx, IndexVec, newtype_index};
use rustc_middle::mir::mono::MonoItem;
use rustc_middle::ty::TyCtxt;

use crate::collector::UsageMap;
use crate::errors;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct StaticNodeIdx(usize);

impl Idx for StaticNodeIdx {
    fn new(idx: usize) -> Self {
        Self(idx)
    }

    fn index(self) -> usize {
        self.0
    }
}

impl From<usize> for StaticNodeIdx {
    fn from(value: usize) -> Self {
        StaticNodeIdx(value)
    }
}

newtype_index! {
    #[derive(Ord, PartialOrd)]
    struct StaticSccIdx {}
}

// Adjacency-list graph for statics using `StaticNodeIdx` as node type.
// We cannot use `DefId` as the node type directly because each node must be
// represented by an index in the range `0..num_nodes`.
struct StaticRefGraph<'a, 'b, 'tcx> {
    // maps from `StaticNodeIdx` to `DefId` and vice versa
    statics: &'a FxIndexSet<DefId>,
    // contains for each `MonoItem` the `MonoItem`s it uses
    used_map: &'b UnordMap<MonoItem<'tcx>, Vec<MonoItem<'tcx>>>,
}

impl<'a, 'b, 'tcx> DirectedGraph for StaticRefGraph<'a, 'b, 'tcx> {
    type Node = StaticNodeIdx;

    fn num_nodes(&self) -> usize {
        self.statics.len()
    }
}

impl<'a, 'b, 'tcx> Successors for StaticRefGraph<'a, 'b, 'tcx> {
    fn successors(&self, node_idx: StaticNodeIdx) -> impl Iterator<Item = StaticNodeIdx> {
        let def_id = self.statics[node_idx.index()];
        self.used_map[&MonoItem::Static(def_id)].iter().filter_map(|&mono_item| match mono_item {
            MonoItem::Static(def_id) => self.statics.get_index_of(&def_id).map(|idx| idx.into()),
            _ => None,
        })
    }
}

pub(super) fn check_static_initializers_are_acyclic<'tcx, 'a, 'b>(
    tcx: TyCtxt<'tcx>,
    mono_items: &'a [MonoItem<'tcx>],
    usage_map: &'b UsageMap<'tcx>,
) {
    // Collect statics
    let statics: FxIndexSet<DefId> = mono_items
        .iter()
        .filter_map(|&mono_item| match mono_item {
            MonoItem::Static(def_id) => Some(def_id),
            _ => None,
        })
        .collect();

    // If we don't have any statics the check is not necessary
    if statics.is_empty() {
        return;
    }
    // Create a subgraph from the mono item graph, which only contains statics
    let graph = StaticRefGraph { statics: &statics, used_map: &usage_map.used_map };
    // Calculate its SCCs
    let sccs: Sccs<StaticNodeIdx, StaticSccIdx> = Sccs::new(&graph);
    // Group statics by SCCs
    let mut nodes_of_sccs: IndexVec<StaticSccIdx, Vec<StaticNodeIdx>> =
        IndexVec::from_elem_n(Vec::new(), sccs.num_sccs());
    for i in graph.iter_nodes() {
        nodes_of_sccs[sccs.scc(i)].push(i);
    }
    let is_cyclic = |nodes_of_scc: &[StaticNodeIdx]| -> bool {
        match nodes_of_scc.len() {
            0 => false,
            1 => graph.successors(nodes_of_scc[0]).any(|x| x == nodes_of_scc[0]),
            2.. => true,
        }
    };
    // Emit errors for all cycles
    for nodes in nodes_of_sccs.iter_mut().filter(|nodes| is_cyclic(nodes)) {
        // We sort the nodes by their Span to have consistent error line numbers
        nodes.sort_by_key(|node| tcx.def_span(statics[node.index()]));

        let head_def = statics[nodes[0].index()];
        let head_span = tcx.def_span(head_def);

        tcx.dcx().emit_err(errors::StaticInitializerCyclic {
            span: head_span,
            labels: nodes.iter().map(|&n| tcx.def_span(statics[n.index()])).collect(),
            head: &tcx.def_path_str(head_def),
            target: &tcx.sess.target.llvm_target,
        });
    }
}
