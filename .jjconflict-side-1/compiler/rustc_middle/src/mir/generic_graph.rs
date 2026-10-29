use gsgdt::{Edge, Graph, Node, NodeStyle};

use crate::mir::*;

/// Convert an MIR function into a gsgdt Graph
pub(crate) fn mir_fn_to_generic_graph<'tcx>(tcx: TyCtxt<'tcx>, body: &Body<'_>) -> Graph {
    let def_id = body.source.def_id();
    let def_name = graphviz_safe_def_name(def_id);
    let graph_name = format!("Mir_{def_name}");
    let dark_mode = tcx.sess.opts.unstable_opts.graphviz_dark_mode;

    // Nodes
    let nodes: Vec<Node> = body
        .basic_blocks
        .iter_enumerated()
        .map(|(block, _)| bb_to_graph_node(block, body, dark_mode))
        .collect();

    // Edges
    let mut edges = Vec::new();
    for (source, _) in body.basic_blocks.iter_enumerated() {
        let def_id = body.source.def_id();
        let terminator = body[source].terminator();
        let labels = terminator.kind.fmt_successor_labels();

        for (target, label) in terminator.successors().zip(labels) {
            let src = node(def_id, source);
            let trg = node(def_id, target);
            edges.push(Edge::new(src, trg, label.to_string()));
        }
    }

    Graph::new(graph_name, nodes, edges)
}

fn bb_to_graph_node(block: BasicBlock, body: &Body<'_>, dark_mode: bool) -> Node {
    let def_id = body.source.def_id();
    let data = &body[block];
    let label = node(def_id, block);

    let (title, bgcolor) = if data.is_cleanup {
        let color = if dark_mode { "royalblue" } else { "lightblue" };
        (format!("{} (cleanup)", block.index()), color)
    } else {
        let color = if dark_mode { "dimgray" } else { "gray" };
        (format!("{}", block.index()), color)
    };

    let style = NodeStyle { title_bg: Some(bgcolor.to_owned()), ..Default::default() };
    let mut stmts: Vec<String> = data.statements.iter().map(|x| format!("{x:?}")).collect();

    // add the terminator to the stmts, gsgdt can print it out separately
    let mut terminator_head = String::new();
    data.terminator().kind.fmt_head(&mut terminator_head).unwrap();
    stmts.push(terminator_head);

    Node::new(stmts, label, title, style)
}

// Must match `[0-9A-Za-z_]*`. This does not appear in the rendered graph, so
// it does not have to be user friendly.
pub fn graphviz_safe_def_name(def_id: DefId) -> String {
    format!("{}_{}", def_id.krate.index(), def_id.index.index(),)
}

fn node(def_id: DefId, block: BasicBlock) -> String {
    format!("bb{}__{}", block.index(), graphviz_safe_def_name(def_id))
}
