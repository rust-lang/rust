// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/*! Generate files suitable for use with [Graphviz](http://www.graphviz.org/)

The `render` function generates output (e.g. an `output.dot` file) for
use with [Graphviz](http://www.graphviz.org/) by walking a labelled
graph. (Graphviz can then automatically lay out the nodes and edges
of the graph, and also optionally render the graph as an image or
other [output formats](
http://www.graphviz.org/content/output-formats), such as SVG.)

Rather than impose some particular graph data structure on clients,
this library exposes two traits that clients can implement on their
own structs before handing them over to the rendering function.

Note: This library does not yet provide access to the full
expressiveness of the [DOT language](
http://www.graphviz.org/doc/info/lang.html). For example, there are
many [attributes](http://www.graphviz.org/content/attrs) related to
providing layout hints (e.g. left-to-right versus top-down, which
algorithm to use, etc). The current intention of this library is to
emit a human-readable .dot file with very regular structure suitable
for easy post-processing.

# Examples

The first example uses a very simple graph representation: a list of
pairs of ints, representing the edges (the node set is implicit).
Each node label is derived directly from the int representing the node,
while the edge labels are all empty strings.

This example also illustrates how to use `MaybeOwnedVector` to return
an owned vector or a borrowed slice as appropriate: we construct the
node vector from scratch, but borrow the edge list (rather than
constructing a copy of all the edges from scratch).

The output from this example renders five nodes, with the first four
forming a diamond-shaped acyclic graph and then pointing to the fifth
which is cyclic.

```rust
use dot = graphviz;
use graphviz::maybe_owned_vec::IntoMaybeOwnedVector;

type Nd = int;
type Ed = (int,int);
struct Edges(Vec<Ed>);

pub fn render_to<W:Writer>(output: &mut W) {
    let edges = Edges(vec!((0,1), (0,2), (1,3), (2,3), (3,4), (4,4)));
    dot::render(&edges, output).unwrap()
}

impl<'a> dot::Labeller<'a, Nd, Ed> for Edges {
    fn graph_id(&'a self) -> dot::Id<'a> { dot::Id::new("example1") }

    fn node_id(&'a self, n: &Nd) -> dot::Id<'a> {
        dot::Id::new(format!("N{}", *n))
    }
}

impl<'a> dot::GraphWalk<'a, Nd, Ed> for Edges {
    fn nodes(&self) -> dot::Nodes<'a,Nd> {
        // (assumes that |N| \approxeq |E|)
        let &Edges(ref v) = self;
        let mut nodes = Vec::with_capacity(v.len());
        for &(s,t) in v.iter() {
            nodes.push(s); nodes.push(t);
        }
        nodes.sort();
        nodes.dedup();
        nodes.into_maybe_owned()
    }

    fn edges(&'a self) -> dot::Edges<'a,Ed> {
        let &Edges(ref edges) = self;
        edges.as_slice().into_maybe_owned()
    }

    fn source(&self, e: &Ed) -> Nd { let &(s,_) = e; s }

    fn target(&self, e: &Ed) -> Nd { let &(_,t) = e; t }
}

# pub fn main() { use std::io::MemWriter; render_to(&mut MemWriter::new()) }
```

```no_run
# pub fn render_to<W:Writer>(output: &mut W) { unimplemented!() }
pub fn main() {
    use std::io::File;
    let mut f = File::create(&Path::new("example1.dot"));
    render_to(&mut f)
}
```

Output from first example (in `example1.dot`):

```ignore
digraph example1 {
    N0[label="N0"];
    N1[label="N1"];
    N2[label="N2"];
    N3[label="N3"];
    N4[label="N4"];
    N0 -> N1[label=""];
    N0 -> N2[label=""];
    N1 -> N3[label=""];
    N2 -> N3[label=""];
    N3 -> N4[label=""];
    N4 -> N4[label=""];
}
```

The second example illustrates using `node_label` and `edge_label` to
add labels to the nodes and edges in the rendered graph. The graph
here carries both `nodes` (the label text to use for rendering a
particular node), and `edges` (again a list of `(source,target)`
indices).

This example also illustrates how to use a type (in this case the edge
type) that shares substructure with the graph: the edge type here is a
direct reference to the `(source,target)` pair stored in the graph's
internal vector (rather than passing around a copy of the pair
itself). Note that this implies that `fn edges(&'a self)` must
construct a fresh `Vec<&'a (uint,uint)>` from the `Vec<(uint,uint)>`
edges stored in `self`.

Since both the set of nodes and the set of edges are always
constructed from scratch via iterators, we use the `collect()` method
from the `Iterator` trait to collect the nodes and edges into freshly
constructed growable `Vec` values (rather use the `into_maybe_owned`
from the `IntoMaybeOwnedVector` trait as was used in the first example
above).

The output from this example renders four nodes that make up the
Hasse-diagram for the subsets of the set `{x, y}`. Each edge is
labelled with the &sube; character (specified using the HTML character
entity `&sube`).

```rust
use dot = graphviz;
use std::str;

type Nd = uint;
type Ed<'a> = &'a (uint, uint);
struct Graph { nodes: Vec<&'static str>, edges: Vec<(uint,uint)> }

pub fn render_to<W:Writer>(output: &mut W) {
    let nodes = vec!("{x,y}","{x}","{y}","{}");
    let edges = vec!((0,1), (0,2), (1,3), (2,3));
    let graph = Graph { nodes: nodes, edges: edges };

    dot::render(&graph, output).unwrap()
}

impl<'a> dot::Labeller<'a, Nd, Ed<'a>> for Graph {
    fn graph_id(&'a self) -> dot::Id<'a> { dot::Id::new("example2") }
    fn node_id(&'a self, n: &Nd) -> dot::Id<'a> {
        dot::Id::new(format!("N{}", n))
    }
    fn node_label<'a>(&'a self, n: &Nd) -> dot::LabelText<'a> {
        dot::LabelStr(str::Slice(self.nodes.get(*n).as_slice()))
    }
    fn edge_label<'a>(&'a self, _: &Ed) -> dot::LabelText<'a> {
        dot::LabelStr(str::Slice("&sube;"))
    }
}

impl<'a> dot::GraphWalk<'a, Nd, Ed<'a>> for Graph {
    fn nodes(&self) -> dot::Nodes<'a,Nd> { range(0,self.nodes.len()).collect() }
    fn edges(&'a self) -> dot::Edges<'a,Ed<'a>> { self.edges.iter().collect() }
    fn source(&self, e: &Ed) -> Nd { let & &(s,_) = e; s }
    fn target(&self, e: &Ed) -> Nd { let & &(_,t) = e; t }
}

# pub fn main() { use std::io::MemWriter; render_to(&mut MemWriter::new()) }
```

```no_run
# pub fn render_to<W:Writer>(output: &mut W) { unimplemented!() }
pub fn main() {
    use std::io::File;
    let mut f = File::create(&Path::new("example2.dot"));
    render_to(&mut f)
}
```

The third example is similar to the second, except now each node and
edge now carries a reference to the string label for each node as well
as that node's index. (This is another illustration of how to share
structure with the graph itself, and why one might want to do so.)

The output from this example is the same as the second example: the
Hasse-diagram for the subsets of the set `{x, y}`.

```rust
use dot = graphviz;
use std::str;

type Nd<'a> = (uint, &'a str);
type Ed<'a> = (Nd<'a>, Nd<'a>);
struct Graph { nodes: Vec<&'static str>, edges: Vec<(uint,uint)> }

pub fn render_to<W:Writer>(output: &mut W) {
    let nodes = vec!("{x,y}","{x}","{y}","{}");
    let edges = vec!((0,1), (0,2), (1,3), (2,3));
    let graph = Graph { nodes: nodes, edges: edges };

    dot::render(&graph, output).unwrap()
}

impl<'a> dot::Labeller<'a, Nd<'a>, Ed<'a>> for Graph {
    fn graph_id(&'a self) -> dot::Id<'a> { dot::Id::new("example3") }
    fn node_id(&'a self, n: &Nd<'a>) -> dot::Id<'a> {
        dot::Id::new(format!("N{:u}", n.val0()))
    }
    fn node_label<'a>(&'a self, n: &Nd<'a>) -> dot::LabelText<'a> {
        let &(i, _) = n;
        dot::LabelStr(str::Slice(self.nodes.get(i).as_slice()))
    }
    fn edge_label<'a>(&'a self, _: &Ed<'a>) -> dot::LabelText<'a> {
        dot::LabelStr(str::Slice("&sube;"))
    }
}

impl<'a> dot::GraphWalk<'a, Nd<'a>, Ed<'a>> for Graph {
    fn nodes(&'a self) -> dot::Nodes<'a,Nd<'a>> {
        self.nodes.iter().map(|s|s.as_slice()).enumerate().collect()
    }
    fn edges(&'a self) -> dot::Edges<'a,Ed<'a>> {
        self.edges.iter()
            .map(|&(i,j)|((i, self.nodes.get(i).as_slice()),
                          (j, self.nodes.get(j).as_slice())))
            .collect()
    }
    fn source(&self, e: &Ed<'a>) -> Nd<'a> { let &(s,_) = e; s }
    fn target(&self, e: &Ed<'a>) -> Nd<'a> { let &(_,t) = e; t }
}

# pub fn main() { use std::io::MemWriter; render_to(&mut MemWriter::new()) }
```

```no_run
# pub fn render_to<W:Writer>(output: &mut W) { unimplemented!() }
pub fn main() {
    use std::io::File;
    let mut f = File::create(&Path::new("example3.dot"));
    render_to(&mut f)
}
```

# References

* [Graphviz](http://www.graphviz.org/)

* [DOT language](http://www.graphviz.org/doc/info/lang.html)

*/

#![crate_id = "graphviz#0.11.0"] // NOTE: remove after stage0
#![crate_name = "graphviz"]
#![experimental]
#![crate_type = "rlib"]
#![crate_type = "dylib"]
#![license = "MIT/ASL2"]
#![doc(html_logo_url = "http://www.rust-lang.org/logos/rust-logo-128x128-blk-v2.png",
       html_favicon_url = "http://www.rust-lang.org/favicon.ico",
       html_root_url = "http://doc.rust-lang.org/0.11.0/")]
#![allow(unused_attribute)] // NOTE: remove after stage0

use std::io;
use std::str;
use self::maybe_owned_vec::MaybeOwnedVector;

pub mod maybe_owned_vec;

/// The text for a graphviz label on a node or edge.
pub enum LabelText<'a> {
    /// This kind of label preserves the text directly as is.
    ///
    /// Occurrences of backslashes (`\`) are escaped, and thus appear
    /// as backslashes in the rendered label.
    LabelStr(str::MaybeOwned<'a>),

    /// This kind of label uses the graphviz label escString type:
    /// http://www.graphviz.org/content/attrs#kescString
    ///
    /// Occurrences of backslashes (`\`) are not escaped; instead they
    /// are interpreted as initiating an escString escape sequence.
    ///
    /// Escape sequences of particular interest: in addition to `\n`
    /// to break a line (centering the line preceding the `\n`), there
    /// are also the escape sequences `\l` which left-justifies the
    /// preceding line and `\r` which right-justifies it.
    EscStr(str::MaybeOwned<'a>),
}

// There is a tension in the design of the labelling API.
//
// For example, I considered making a `Labeller<T>` trait that
// provides labels for `T`, and then making the graph type `G`
// implement `Labeller<Node>` and `Labeller<Edge>`. However, this is
// not possible without functional dependencies. (One could work
// around that, but I did not explore that avenue heavily.)
//
// Another approach that I actually used for a while was to make a
// `Label<Context>` trait that is implemented by the client-specific
// Node and Edge types (as well as an implementation on Graph itself
// for the overall name for the graph). The main disadvantage of this
// second approach (compared to having the `G` type parameter
// implement a Labelling service) that I have encountered is that it
// makes it impossible to use types outside of the current crate
// directly as Nodes/Edges; you need to wrap them in newtype'd
// structs. See e.g. the `No` and `Ed` structs in the examples. (In
// practice clients using a graph in some other crate would need to
// provide some sort of adapter shim over the graph anyway to
// interface with this library).
//
// Another approach would be to make a single `Labeller<N,E>` trait
// that provides three methods (graph_label, node_label, edge_label),
// and then make `G` implement `Labeller<N,E>`. At first this did not
// appeal to me, since I had thought I would need separate methods on
// each data variant for dot-internal identifiers versus user-visible
// labels. However, the identifier/label distinction only arises for
// nodes; graphs themselves only have identifiers, and edges only have
// labels.
//
// So in the end I decided to use the third approach described above.

/// `Id` is a Graphviz `ID`.
pub struct Id<'a> {
    name: str::MaybeOwned<'a>,
}

impl<'a> Id<'a> {
    /// Creates an `Id` named `name`.
    ///
    /// The caller must ensure that the input conforms to an
    /// identifier format: it must be a non-empty string made up of
    /// alphanumeric or underscore characters, not beginning with a
    /// digit (i.e. the regular expression `[a-zA-Z_][a-zA-Z_0-9]*`).
    ///
    /// (Note: this format is a strict subset of the `ID` format
    /// defined by the DOT language.  This function may change in the
    /// future to accept a broader subset, or the entirety, of DOT's
    /// `ID` format.)
    pub fn new<Name:str::IntoMaybeOwned<'a>>(name: Name) -> Id<'a> {
        let name = name.into_maybe_owned();
        {
            let mut chars = name.as_slice().chars();
            assert!(is_letter_or_underscore(chars.next().unwrap()));
            assert!(chars.all(is_constituent));
        }
        return Id{ name: name };

        fn is_letter_or_underscore(c: char) -> bool {
            in_range('a', c, 'z') || in_range('A', c, 'Z') || c == '_'
        }
        fn is_constituent(c: char) -> bool {
            is_letter_or_underscore(c) || in_range('0', c, '9')
        }
        fn in_range(low: char, c: char, high: char) -> bool {
            low as uint <= c as uint && c as uint <= high as uint
        }
    }

    pub fn as_slice(&'a self) -> &'a str {
        self.name.as_slice()
    }

    pub fn name(self) -> str::MaybeOwned<'a> {
        self.name
    }
}

/// Each instance of a type that implements `Label<C>` maps to a
/// unique identifier with respect to `C`, which is used to identify
/// it in the generated .dot file. They can also provide more
/// elaborate (and non-unique) label text that is used in the graphviz
/// rendered output.

/// The graph instance is responsible for providing the DOT compatible
/// identifiers for the nodes and (optionally) rendered labels for the nodes and
/// edges, as well as an identifier for the graph itself.
pub trait Labeller<'a,N,E> {
    /// Must return a DOT compatible identifier naming the graph.
    fn graph_id(&'a self) -> Id<'a>;

    /// Maps `n` to a unique identifier with respect to `self`. The
    /// implementer is responsible for ensuring that the returned name
    /// is a valid DOT identifier.
    fn node_id(&'a self, n: &N) -> Id<'a>;

    /// Maps `n` to a label that will be used in the rendered output.
    /// The label need not be unique, and may be the empty string; the
    /// default is just the output from `node_id`.
    fn node_label(&'a self, n: &N) -> LabelText<'a> {
        LabelStr(self.node_id(n).name)
    }

    /// Maps `e` to a label that will be used in the rendered output.
    /// The label need not be unique, and may be the empty string; the
    /// default is in fact the empty string.
    fn edge_label(&'a self, e: &E) -> LabelText<'a> {
        let _ignored = e;
        LabelStr(str::Slice(""))
    }
}

impl<'a> LabelText<'a> {
    fn escape_char(c: char, f: |char|) {
        match c {
            // not escaping \\, since Graphviz escString needs to
            // interpret backslashes; see EscStr above.
            '\\' => f(c),
            _ => c.escape_default(f)
        }
    }
    fn escape_str(s: &str) -> String {
        let mut out = String::with_capacity(s.len());
        for c in s.chars() {
            LabelText::escape_char(c, |c| out.push_char(c));
        }
        out
    }

    /// Renders text as string suitable for a label in a .dot file.
    pub fn escape(&self) -> String {
        match self {
            &LabelStr(ref s) => s.as_slice().escape_default().to_string(),
            &EscStr(ref s) => LabelText::escape_str(s.as_slice()).to_string(),
        }
    }
}

pub type Nodes<'a,N> = MaybeOwnedVector<'a,N>;
pub type Edges<'a,E> = MaybeOwnedVector<'a,E>;

// (The type parameters in GraphWalk should be associated items,
// when/if Rust supports such.)

/// GraphWalk is an abstraction over a directed graph = (nodes,edges)
/// made up of node handles `N` and edge handles `E`, where each `E`
/// can be mapped to its source and target nodes.
///
/// The lifetime parameter `'a` is exposed in this trait (rather than
/// introduced as a generic parameter on each method declaration) so
/// that a client impl can choose `N` and `E` that have substructure
/// that is bound by the self lifetime `'a`.
///
/// The `nodes` and `edges` method each return instantiations of
/// `MaybeOwnedVector` to leave implementers the freedom to create
/// entirely new vectors or to pass back slices into internally owned
/// vectors.
pub trait GraphWalk<'a, N, E> {
    /// Returns all the nodes in this graph.
    fn nodes(&'a self) -> Nodes<'a, N>;
    /// Returns all of the edges in this graph.
    fn edges(&'a self) -> Edges<'a, E>;
    /// The source node for `edge`.
    fn source(&'a self, edge: &E) -> N;
    /// The target node for `edge`.
    fn target(&'a self, edge: &E) -> N;
}

/// Renders directed graph `g` into the writer `w` in DOT syntax.
/// (Main entry point for the library.)
pub fn render<'a, N, E, G:Labeller<'a,N,E>+GraphWalk<'a,N,E>, W:Writer>(
              g: &'a G,
              w: &mut W) -> io::IoResult<()>
{
    fn writeln<W:Writer>(w: &mut W, arg: &[&str]) -> io::IoResult<()> {
        for &s in arg.iter() { try!(w.write_str(s)); }
        w.write_char('\n')
    }

    fn indent<W:Writer>(w: &mut W) -> io::IoResult<()> {
        w.write_str("    ")
    }

    try!(writeln(w, ["digraph ", g.graph_id().as_slice(), " {"]));
    for n in g.nodes().iter() {
        try!(indent(w));
        let id = g.node_id(n);
        let escaped = g.node_label(n).escape();
        try!(writeln(w, [id.as_slice(),
                         "[label=\"", escaped.as_slice(), "\"];"]));
    }

    for e in g.edges().iter() {
        let escaped_label = g.edge_label(e).escape();
        try!(indent(w));
        let source = g.source(e);
        let target = g.target(e);
        let source_id = g.node_id(&source);
        let target_id = g.node_id(&target);
        try!(writeln(w, [source_id.as_slice(), " -> ", target_id.as_slice(),
                         "[label=\"", escaped_label.as_slice(), "\"];"]));
    }

    writeln(w, ["}"])
}

#[cfg(test)]
mod tests {
    use super::{Id, LabelText, LabelStr, EscStr, Labeller};
    use super::{Nodes, Edges, GraphWalk, render};
    use std::io::{MemWriter, BufReader, IoResult};
    use std::str;

    /// each node is an index in a vector in the graph.
    type Node = uint;
    struct Edge {
        from: uint, to: uint, label: &'static str
    }

    fn Edge(from: uint, to: uint, label: &'static str) -> Edge {
        Edge { from: from, to: to, label: label }
    }

    struct LabelledGraph {
        /// The name for this graph. Used for labelling generated `digraph`.
        name: &'static str,

        /// Each node is an index into `node_labels`; these labels are
        /// used as the label text for each node. (The node *names*,
        /// which are unique identifiers, are derived from their index
        /// in this array.)
        ///
        /// If a node maps to None here, then just use its name as its
        /// text.
        node_labels: Vec<Option<&'static str>>,

        /// Each edge relates a from-index to a to-index along with a
        /// label; `edges` collects them.
        edges: Vec<Edge>,
    }

    // A simple wrapper around LabelledGraph that forces the labels to
    // be emitted as EscStr.
    struct LabelledGraphWithEscStrs {
        graph: LabelledGraph
    }

    enum NodeLabels<L> {
        AllNodesLabelled(Vec<L>),
        UnlabelledNodes(uint),
        SomeNodesLabelled(Vec<Option<L>>),
    }

    type Trivial = NodeLabels<&'static str>;

    impl NodeLabels<&'static str> {
        fn to_opt_strs(self) -> Vec<Option<&'static str>> {
            match self {
                UnlabelledNodes(len)
                    => Vec::from_elem(len, None).move_iter().collect(),
                AllNodesLabelled(lbls)
                    => lbls.move_iter().map(
                        |l|Some(l)).collect(),
                SomeNodesLabelled(lbls)
                    => lbls.move_iter().collect(),
            }
        }
    }

    impl LabelledGraph {
        fn new(name: &'static str,
               node_labels: Trivial,
               edges: Vec<Edge>) -> LabelledGraph {
            LabelledGraph {
                name: name,
                node_labels: node_labels.to_opt_strs(),
                edges: edges
            }
        }
    }

    impl LabelledGraphWithEscStrs {
        fn new(name: &'static str,
               node_labels: Trivial,
               edges: Vec<Edge>) -> LabelledGraphWithEscStrs {
            LabelledGraphWithEscStrs {
                graph: LabelledGraph::new(name, node_labels, edges)
            }
        }
    }

    fn id_name<'a>(n: &Node) -> Id<'a> {
        Id::new(format!("N{:u}", *n))
    }

    impl<'a> Labeller<'a, Node, &'a Edge> for LabelledGraph {
        fn graph_id(&'a self) -> Id<'a> {
            Id::new(self.name.as_slice())
        }
        fn node_id(&'a self, n: &Node) -> Id<'a> {
            id_name(n)
        }
        fn node_label(&'a self, n: &Node) -> LabelText<'a> {
            match self.node_labels.get(*n) {
                &Some(ref l) => LabelStr(str::Slice(l.as_slice())),
                &None        => LabelStr(id_name(n).name()),
            }
        }
        fn edge_label(&'a self, e: & &'a Edge) -> LabelText<'a> {
            LabelStr(str::Slice(e.label.as_slice()))
        }
    }

    impl<'a> Labeller<'a, Node, &'a Edge> for LabelledGraphWithEscStrs {
        fn graph_id(&'a self) -> Id<'a> { self.graph.graph_id() }
        fn node_id(&'a self, n: &Node) -> Id<'a> { self.graph.node_id(n) }
        fn node_label(&'a self, n: &Node) -> LabelText<'a> {
            match self.graph.node_label(n) {
                LabelStr(s) | EscStr(s) => EscStr(s),
            }
        }
        fn edge_label(&'a self, e: & &'a Edge) -> LabelText<'a> {
            match self.graph.edge_label(e) {
                LabelStr(s) | EscStr(s) => EscStr(s),
            }
        }
    }

    impl<'a> GraphWalk<'a, Node, &'a Edge> for LabelledGraph {
        fn nodes(&'a self) -> Nodes<'a,Node> {
            range(0u, self.node_labels.len()).collect()
        }
        fn edges(&'a self) -> Edges<'a,&'a Edge> {
            self.edges.iter().collect()
        }
        fn source(&'a self, edge: & &'a Edge) -> Node {
            edge.from
        }
        fn target(&'a self, edge: & &'a Edge) -> Node {
            edge.to
        }
    }

    impl<'a> GraphWalk<'a, Node, &'a Edge> for LabelledGraphWithEscStrs {
        fn nodes(&'a self) -> Nodes<'a,Node> {
            self.graph.nodes()
        }
        fn edges(&'a self) -> Edges<'a,&'a Edge> {
            self.graph.edges()
        }
        fn source(&'a self, edge: & &'a Edge) -> Node {
            edge.from
        }
        fn target(&'a self, edge: & &'a Edge) -> Node {
            edge.to
        }
    }

    fn test_input(g: LabelledGraph) -> IoResult<String> {
        let mut writer = MemWriter::new();
        render(&g, &mut writer).unwrap();
        let mut r = BufReader::new(writer.get_ref());
        match r.read_to_string() {
            Ok(string) => Ok(string.to_string()),
            Err(err) => Err(err),
        }
    }

    // All of the tests use raw-strings as the format for the expected outputs,
    // so that you can cut-and-paste the content into a .dot file yourself to
    // see what the graphviz visualizer would produce.

    #[test]
    fn empty_graph() {
        let labels : Trivial = UnlabelledNodes(0);
        let r = test_input(LabelledGraph::new("empty_graph", labels, vec!()));
        assert_eq!(r.unwrap().as_slice(),
r#"digraph empty_graph {
}
"#);
    }

    #[test]
    fn single_node() {
        let labels : Trivial = UnlabelledNodes(1);
        let r = test_input(LabelledGraph::new("single_node", labels, vec!()));
        assert_eq!(r.unwrap().as_slice(),
r#"digraph single_node {
    N0[label="N0"];
}
"#);
    }

    #[test]
    fn single_edge() {
        let labels : Trivial = UnlabelledNodes(2);
        let result = test_input(LabelledGraph::new("single_edge", labels,
                                                   vec!(Edge(0, 1, "E"))));
        assert_eq!(result.unwrap().as_slice(),
r#"digraph single_edge {
    N0[label="N0"];
    N1[label="N1"];
    N0 -> N1[label="E"];
}
"#);
    }

    #[test]
    fn single_cyclic_node() {
        let labels : Trivial = UnlabelledNodes(1);
        let r = test_input(LabelledGraph::new("single_cyclic_node", labels,
                                              vec!(Edge(0, 0, "E"))));
        assert_eq!(r.unwrap().as_slice(),
r#"digraph single_cyclic_node {
    N0[label="N0"];
    N0 -> N0[label="E"];
}
"#);
    }

    #[test]
    fn hasse_diagram() {
        let labels = AllNodesLabelled(vec!("{x,y}", "{x}", "{y}", "{}"));
        let r = test_input(LabelledGraph::new(
            "hasse_diagram", labels,
            vec!(Edge(0, 1, ""), Edge(0, 2, ""),
                 Edge(1, 3, ""), Edge(2, 3, ""))));
        assert_eq!(r.unwrap().as_slice(),
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
"#);
    }

    #[test]
    fn left_aligned_text() {
        let labels = AllNodesLabelled(vec!(
            "if test {\
           \\l    branch1\
           \\l} else {\
           \\l    branch2\
           \\l}\
           \\lafterward\
           \\l",
            "branch1",
            "branch2",
            "afterward"));

        let mut writer = MemWriter::new();

        let g = LabelledGraphWithEscStrs::new(
            "syntax_tree", labels,
            vec!(Edge(0, 1, "then"), Edge(0, 2, "else"),
                 Edge(1, 3, ";"),    Edge(2, 3, ";"   )));

        render(&g, &mut writer).unwrap();
        let mut r = BufReader::new(writer.get_ref());
        let r = r.read_to_string();

        assert_eq!(r.unwrap().as_slice(),
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
"#);
    }
}
