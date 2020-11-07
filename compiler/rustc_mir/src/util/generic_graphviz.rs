use rustc_data_structures::graph::{self, iterate};
use rustc_graphviz as dot;
use rustc_middle::ty::TyCtxt;
use std::io::{self, Write};

pub struct GraphvizWriter<
    'a,
    G: graph::DirectedGraph + graph::WithSuccessors + graph::WithStartNode + graph::WithNumNodes,
    NodeContentFn: Fn(<G as rustc_data_structures::graph::DirectedGraph>::Node) -> Vec<String>,
    EdgeLabelsFn: Fn(<G as rustc_data_structures::graph::DirectedGraph>::Node) -> Vec<String>,
> {
    graph: &'a G,
    is_subgraph: bool,
    graphviz_name: String,
    graph_label: Option<String>,
    node_content_fn: NodeContentFn,
    edge_labels_fn: EdgeLabelsFn,
}

impl<
    'a,
    G: graph::DirectedGraph + graph::WithSuccessors + graph::WithStartNode + graph::WithNumNodes,
    NodeContentFn: Fn(<G as rustc_data_structures::graph::DirectedGraph>::Node) -> Vec<String>,
    EdgeLabelsFn: Fn(<G as rustc_data_structures::graph::DirectedGraph>::Node) -> Vec<String>,
> GraphvizWriter<'a, G, NodeContentFn, EdgeLabelsFn>
{
    pub fn new(
        graph: &'a G,
        graphviz_name: &str,
        node_content_fn: NodeContentFn,
        edge_labels_fn: EdgeLabelsFn,
    ) -> Self {
        Self {
            graph,
            is_subgraph: false,
            graphviz_name: graphviz_name.to_owned(),
            graph_label: None,
            node_content_fn,
            edge_labels_fn,
        }
    }

    pub fn new_subgraph(
        graph: &'a G,
        graphviz_name: &str,
        node_content_fn: NodeContentFn,
        edge_labels_fn: EdgeLabelsFn,
    ) -> Self {
        Self {
            graph,
            is_subgraph: true,
            graphviz_name: graphviz_name.to_owned(),
            graph_label: None,
            node_content_fn,
            edge_labels_fn,
        }
    }

    pub fn set_graph_label(&mut self, graph_label: &str) {
        self.graph_label = Some(graph_label.to_owned());
    }

    /// Write a graphviz DOT of the graph
    pub fn write_graphviz<'tcx, W>(&self, tcx: TyCtxt<'tcx>, w: &mut W) -> io::Result<()>
    where
        W: Write,
    {
        let kind = if self.is_subgraph { "subgraph" } else { "digraph" };
        let cluster = if self.is_subgraph { "cluster_" } else { "" }; // Print border around graph
        // FIXME(richkadel): If/when migrating the MIR graphviz to this generic implementation,
        // prepend "Mir_" to the graphviz_safe_def_name(def_id)
        writeln!(w, "{} {}{} {{", kind, cluster, self.graphviz_name)?;

        // Global graph properties
        let font = format!(r#"fontname="{}""#, tcx.sess.opts.debugging_opts.graphviz_font);
        let mut graph_attrs = vec![&font[..]];
        let mut content_attrs = vec![&font[..]];

        let dark_mode = tcx.sess.opts.debugging_opts.graphviz_dark_mode;
        if dark_mode {
            graph_attrs.push(r#"bgcolor="black""#);
            graph_attrs.push(r#"fontcolor="white""#);
            content_attrs.push(r#"color="white""#);
            content_attrs.push(r#"fontcolor="white""#);
        }

        writeln!(w, r#"    graph [{}];"#, graph_attrs.join(" "))?;
        let content_attrs_str = content_attrs.join(" ");
        writeln!(w, r#"    node [{}];"#, content_attrs_str)?;
        writeln!(w, r#"    edge [{}];"#, content_attrs_str)?;

        // Graph label
        if let Some(graph_label) = &self.graph_label {
            self.write_graph_label(graph_label, w)?;
        }

        // Nodes
        for node in iterate::post_order_from(self.graph, self.graph.start_node()) {
            self.write_node(node, dark_mode, w)?;
        }

        // Edges
        for source in iterate::post_order_from(self.graph, self.graph.start_node()) {
            self.write_edges(source, w)?;
        }
        writeln!(w, "}}")
    }

    /// Write a graphviz DOT node for the given node.
    pub fn write_node<W>(&self, node: G::Node, dark_mode: bool, w: &mut W) -> io::Result<()>
    where
        W: Write,
    {
        // Start a new node with the label to follow, in one of DOT's pseudo-HTML tables.
        write!(w, r#"    {} [shape="none", label=<"#, self.node(node))?;

        write!(w, r#"<table border="0" cellborder="1" cellspacing="0">"#)?;

        // FIXME(richkadel): Need generic way to know if node header should have a different color
        // let (blk, bgcolor) = if data.is_cleanup {
        //    (format!("{:?} (cleanup)", node), "lightblue")
        // } else {
        //     let color = if dark_mode { "dimgray" } else { "gray" };
        //     (format!("{:?}", node), color)
        // };
        let color = if dark_mode { "dimgray" } else { "gray" };
        let (blk, bgcolor) = (format!("{:?}", node), color);
        write!(
            w,
            r#"<tr><td bgcolor="{bgcolor}" {attrs} colspan="{colspan}">{blk}</td></tr>"#,
            attrs = r#"align="center""#,
            colspan = 1,
            blk = blk,
            bgcolor = bgcolor
        )?;

        for section in (self.node_content_fn)(node) {
            write!(
                w,
                r#"<tr><td align="left" balign="left">{}</td></tr>"#,
                dot::escape_html(&section).replace("\n", "<br/>")
            )?;
        }

        // Close the table
        write!(w, "</table>")?;

        // Close the node label and the node itself.
        writeln!(w, ">];")
    }

    /// Write graphviz DOT edges with labels between the given node and all of its successors.
    fn write_edges<W>(&self, source: G::Node, w: &mut W) -> io::Result<()>
    where
        W: Write,
    {
        let edge_labels = (self.edge_labels_fn)(source);
        for (index, target) in self.graph.successors(source).enumerate() {
            let src = self.node(source);
            let trg = self.node(target);
            let escaped_edge_label = if let Some(edge_label) = edge_labels.get(index) {
                dot::escape_html(edge_label).replace("\n", r#"<br align="left"/>"#)
            } else {
                "".to_owned()
            };
            writeln!(w, r#"    {} -> {} [label=<{}>];"#, src, trg, escaped_edge_label)?;
        }
        Ok(())
    }

    /// Write the graphviz DOT label for the overall graph. This is essentially a block of text that
    /// will appear below the graph.
    fn write_graph_label<W>(&self, label: &str, w: &mut W) -> io::Result<()>
    where
        W: Write,
    {
        let lines = label.split('\n').map(|s| dot::escape_html(s)).collect::<Vec<_>>();
        let escaped_label = lines.join(r#"<br align="left"/>"#);
        writeln!(w, r#"    label=<<br/><br/>{}<br align="left"/><br/><br/><br/>>;"#, escaped_label)
    }

    fn node(&self, node: G::Node) -> String {
        format!("{:?}__{}", node, self.graphviz_name)
    }
}
