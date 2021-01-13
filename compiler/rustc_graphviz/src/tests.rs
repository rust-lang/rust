use super::LabelText::{self, EscStr, HtmlStr, LabelStr};
use super::{render, Edges, GraphWalk, Id, Labeller, Nodes, Style};
use std::io;
use std::io::prelude::*;
use NodeLabels::*;

/// each node is an index in a vector in the graph.
type Node = usize;
struct Edge {
    from: usize,
    to: usize,
    label: &'static str,
    style: Style,
}

fn edge(from: usize, to: usize, label: &'static str, style: Style) -> Edge {
    Edge { from, to, label, style }
}

struct LabelledGraph {
    /// The name for this graph. Used for labeling generated `digraph`.
    name: &'static str,

    /// Each node is an index into `node_labels`; these labels are
    /// used as the label text for each node. (The node *names*,
    /// which are unique identifiers, are derived from their index
    /// in this array.)
    ///
    /// If a node maps to None here, then just use its name as its
    /// text.
    node_labels: Vec<Option<&'static str>>,

    node_styles: Vec<Style>,

    /// Each edge relates a from-index to a to-index along with a
    /// label; `edges` collects them.
    edges: Vec<Edge>,
}

// A simple wrapper around LabelledGraph that forces the labels to
// be emitted as EscStr.
struct LabelledGraphWithEscStrs {
    graph: LabelledGraph,
}

enum NodeLabels<L> {
    AllNodesLabelled(Vec<L>),
    UnlabelledNodes(usize),
    SomeNodesLabelled(Vec<Option<L>>),
}

type Trivial = NodeLabels<&'static str>;

impl NodeLabels<&'static str> {
    fn to_opt_strs(self) -> Vec<Option<&'static str>> {
        match self {
            UnlabelledNodes(len) => vec![None; len],
            AllNodesLabelled(lbls) => lbls.into_iter().map(Some).collect(),
            SomeNodesLabelled(lbls) => lbls.into_iter().collect(),
        }
    }

    fn len(&self) -> usize {
        match self {
            &UnlabelledNodes(len) => len,
            &AllNodesLabelled(ref lbls) => lbls.len(),
            &SomeNodesLabelled(ref lbls) => lbls.len(),
        }
    }
}

impl LabelledGraph {
    fn new(
        name: &'static str,
        node_labels: Trivial,
        edges: Vec<Edge>,
        node_styles: Option<Vec<Style>>,
    ) -> LabelledGraph {
        let count = node_labels.len();
        LabelledGraph {
            name,
            node_labels: node_labels.to_opt_strs(),
            edges,
            node_styles: match node_styles {
                Some(nodes) => nodes,
                None => vec![Style::None; count],
            },
        }
    }
}

impl LabelledGraphWithEscStrs {
    fn new(name: &'static str, node_labels: Trivial, edges: Vec<Edge>) -> LabelledGraphWithEscStrs {
        LabelledGraphWithEscStrs { graph: LabelledGraph::new(name, node_labels, edges, None) }
    }
}

fn id_name<'a>(n: &Node) -> Id<'a> {
    Id::new(format!("N{}", *n)).unwrap()
}

impl<'a> Labeller<'a> for LabelledGraph {
    type Node = Node;
    type Edge = &'a Edge;
    fn graph_id(&'a self) -> Id<'a> {
        Id::new(self.name).unwrap()
    }
    fn node_id(&'a self, n: &Node) -> Id<'a> {
        id_name(n)
    }
    fn node_label(&'a self, n: &Node) -> LabelText<'a> {
        match self.node_labels[*n] {
            Some(l) => LabelStr(l.into()),
            None => LabelStr(id_name(n).name()),
        }
    }
    fn edge_label(&'a self, e: &&'a Edge) -> LabelText<'a> {
        LabelStr(e.label.into())
    }
    fn node_style(&'a self, n: &Node) -> Style {
        self.node_styles[*n]
    }
    fn edge_style(&'a self, e: &&'a Edge) -> Style {
        e.style
    }
}

impl<'a> Labeller<'a> for LabelledGraphWithEscStrs {
    type Node = Node;
    type Edge = &'a Edge;
    fn graph_id(&'a self) -> Id<'a> {
        self.graph.graph_id()
    }
    fn node_id(&'a self, n: &Node) -> Id<'a> {
        self.graph.node_id(n)
    }
    fn node_label(&'a self, n: &Node) -> LabelText<'a> {
        match self.graph.node_label(n) {
            LabelStr(s) | EscStr(s) | HtmlStr(s) => EscStr(s),
        }
    }
    fn edge_label(&'a self, e: &&'a Edge) -> LabelText<'a> {
        match self.graph.edge_label(e) {
            LabelStr(s) | EscStr(s) | HtmlStr(s) => EscStr(s),
        }
    }
}

impl<'a> GraphWalk<'a> for LabelledGraph {
    type Node = Node;
    type Edge = &'a Edge;
    fn nodes(&'a self) -> Nodes<'a, Node> {
        (0..self.node_labels.len()).collect()
    }
    fn edges(&'a self) -> Edges<'a, &'a Edge> {
        self.edges.iter().collect()
    }
    fn source(&'a self, edge: &&'a Edge) -> Node {
        edge.from
    }
    fn target(&'a self, edge: &&'a Edge) -> Node {
        edge.to
    }
}

impl<'a> GraphWalk<'a> for LabelledGraphWithEscStrs {
    type Node = Node;
    type Edge = &'a Edge;
    fn nodes(&'a self) -> Nodes<'a, Node> {
        self.graph.nodes()
    }
    fn edges(&'a self) -> Edges<'a, &'a Edge> {
        self.graph.edges()
    }
    fn source(&'a self, edge: &&'a Edge) -> Node {
        edge.from
    }
    fn target(&'a self, edge: &&'a Edge) -> Node {
        edge.to
    }
}

fn test_input(g: LabelledGraph) -> io::Result<String> {
    let mut writer = Vec::new();
    render(&g, &mut writer).unwrap();
    let mut s = String::new();
    Read::read_to_string(&mut &*writer, &mut s)?;
    Ok(s)
}

// All of the tests use raw-strings as the format for the expected outputs,
// so that you can cut-and-paste the content into a .dot file yourself to
// see what the graphviz visualizer would produce.

#[test]
fn empty_graph() {
    let labels: Trivial = UnlabelledNodes(0);
    let r = test_input(LabelledGraph::new("empty_graph", labels, vec![], None));
    assert_eq!(
        r.unwrap(),
        r#"digraph empty_graph {
}
"#
    );
}

#[test]
fn single_node() {
    let labels: Trivial = UnlabelledNodes(1);
    let r = test_input(LabelledGraph::new("single_node", labels, vec![], None));
    assert_eq!(
        r.unwrap(),
        r#"digraph single_node {
    N0[label="N0"];
}
"#
    );
}

#[test]
fn single_node_with_style() {
    let labels: Trivial = UnlabelledNodes(1);
    let styles = Some(vec![Style::Dashed]);
    let r = test_input(LabelledGraph::new("single_node", labels, vec![], styles));
    assert_eq!(
        r.unwrap(),
        r#"digraph single_node {
    N0[label="N0"][style="dashed"];
}
"#
    );
}

#[test]
fn single_edge() {
    let labels: Trivial = UnlabelledNodes(2);
    let result = test_input(LabelledGraph::new(
        "single_edge",
        labels,
        vec![edge(0, 1, "E", Style::None)],
        None,
    ));
    assert_eq!(
        result.unwrap(),
        r#"digraph single_edge {
    N0[label="N0"];
    N1[label="N1"];
    N0 -> N1[label="E"];
}
"#
    );
}

#[test]
fn single_edge_with_style() {
    let labels: Trivial = UnlabelledNodes(2);
    let result = test_input(LabelledGraph::new(
        "single_edge",
        labels,
        vec![edge(0, 1, "E", Style::Bold)],
        None,
    ));
    assert_eq!(
        result.unwrap(),
        r#"digraph single_edge {
    N0[label="N0"];
    N1[label="N1"];
    N0 -> N1[label="E"][style="bold"];
}
"#
    );
}

#[test]
fn test_some_labelled() {
    let labels: Trivial = SomeNodesLabelled(vec![Some("A"), None]);
    let styles = Some(vec![Style::None, Style::Dotted]);
    let result = test_input(LabelledGraph::new(
        "test_some_labelled",
        labels,
        vec![edge(0, 1, "A-1", Style::None)],
        styles,
    ));
    assert_eq!(
        result.unwrap(),
        r#"digraph test_some_labelled {
    N0[label="A"];
    N1[label="N1"][style="dotted"];
    N0 -> N1[label="A-1"];
}
"#
    );
}

#[test]
fn single_cyclic_node() {
    let labels: Trivial = UnlabelledNodes(1);
    let r = test_input(LabelledGraph::new(
        "single_cyclic_node",
        labels,
        vec![edge(0, 0, "E", Style::None)],
        None,
    ));
    assert_eq!(
        r.unwrap(),
        r#"digraph single_cyclic_node {
    N0[label="N0"];
    N0 -> N0[label="E"];
}
"#
    );
}

#[test]
fn hasse_diagram() {
    let labels = AllNodesLabelled(vec!["{x,y}", "{x}", "{y}", "{}"]);
    let r = test_input(LabelledGraph::new(
        "hasse_diagram",
        labels,
        vec![
            edge(0, 1, "", Style::None),
            edge(0, 2, "", Style::None),
            edge(1, 3, "", Style::None),
            edge(2, 3, "", Style::None),
        ],
        None,
    ));
    assert_eq!(
        r.unwrap(),
        r#"digraph hasse_diagram {
    N0[label="{x,y}"];
    N1[label="{x}"];
    N2[label="{y}"];
    N3[label="{}"];
    N0 -> N1[label=""];
    N0 -> N2[label=""];
    N1 -> N3[label=""];
    N2 -> N3[label=""];
}
"#
    );
}

#[test]
fn left_aligned_text() {
    let labels = AllNodesLabelled(vec![
        "if test {\
       \\l    branch1\
       \\l} else {\
       \\l    branch2\
       \\l}\
       \\lafterward\
       \\l",
        "branch1",
        "branch2",
        "afterward",
    ]);

    let mut writer = Vec::new();

    let g = LabelledGraphWithEscStrs::new(
        "syntax_tree",
        labels,
        vec![
            edge(0, 1, "then", Style::None),
            edge(0, 2, "else", Style::None),
            edge(1, 3, ";", Style::None),
            edge(2, 3, ";", Style::None),
        ],
    );

    render(&g, &mut writer).unwrap();
    let mut r = String::new();
    Read::read_to_string(&mut &*writer, &mut r).unwrap();

    assert_eq!(
        r,
        r#"digraph syntax_tree {
    N0[label="if test {\l    branch1\l} else {\l    branch2\l}\lafterward\l"];
    N1[label="branch1"];
    N2[label="branch2"];
    N3[label="afterward"];
    N0 -> N1[label="then"];
    N0 -> N2[label="else"];
    N1 -> N3[label=";"];
    N2 -> N3[label=";"];
}
"#
    );
}

#[test]
fn simple_id_construction() {
    let id1 = Id::new("hello");
    match id1 {
        Ok(_) => {}
        Err(..) => panic!("'hello' is not a valid value for id anymore"),
    }
}

#[test]
fn badly_formatted_id() {
    let id2 = Id::new("Weird { struct : ure } !!!");
    match id2 {
        Ok(_) => panic!("graphviz id suddenly allows spaces, brackets and stuff"),
        Err(..) => {}
    }
}
