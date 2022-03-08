//! Implementation of GraphWalk for DropRanges so we can visualize the control
//! flow graph when needed for debugging.

use rustc_graphviz as dot;

use super::{DropRangesBuilder, PostOrderId};

/// Writes the CFG for DropRangesBuilder to a .dot file for visualization.
///
/// It is not normally called, but is kept around to easily add debugging
/// code when needed.
#[allow(dead_code)]
pub(super) fn write_graph_to_file(drop_ranges: &DropRangesBuilder, filename: &str) {
    dot::render(drop_ranges, &mut std::fs::File::create(filename).unwrap()).unwrap();
}

impl<'a> dot::GraphWalk<'a> for DropRangesBuilder {
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
                    vec![(i, i + 1)]
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

impl<'a> dot::Labeller<'a> for DropRangesBuilder {
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
                    .find(|(_hir_id, &post_order_id)| post_order_id == *n)
                    .map_or("<unknown>".into(), |(hir_id, _)| format!(
                        "{}",
                        hir_id.local_id.index()
                    ))
            )
            .into(),
        )
    }
}
