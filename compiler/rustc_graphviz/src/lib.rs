//! Generate files suitable for use with [Graphviz](https://www.graphviz.org/)
//!
//! The `render` function generates output (e.g., an `output.dot` file) for
//! use with [Graphviz](https://www.graphviz.org/) by walking a labeled
//! graph. (Graphviz can then automatically lay out the nodes and edges
//! of the graph, and also optionally render the graph as an image or
//! other [output formats](https://www.graphviz.org/docs/outputs), such as SVG.)
//!
//! Rather than impose some particular graph data structure on clients,
//! this library exposes two traits that clients can implement on their
//! own structs before handing them over to the rendering function.
//!
//! Note: This library does not yet provide access to the full
//! expressiveness of the [DOT language](https://www.graphviz.org/doc/info/lang.html).
//! For example, there are many [attributes](https://www.graphviz.org/doc/info/attrs.html)
//! related to providing layout hints (e.g., left-to-right versus top-down, which
//! algorithm to use, etc). The current intention of this library is to
//! emit a human-readable .dot file with very regular structure suitable
//! for easy post-processing.
//!
//! # Examples
//!
//! The first example uses a very simple graph representation: a list of
//! pairs of ints, representing the edges (the node set is implicit).
//! Each node label is derived directly from the int representing the node,
//! while the edge labels are all empty strings.
//!
//! This example also illustrates how to use `Cow<[T]>` to return
//! an owned vector or a borrowed slice as appropriate: we construct the
//! node vector from scratch, but borrow the edge list (rather than
//! constructing a copy of all the edges from scratch).
//!
//! The output from this example renders five nodes, with the first four
//! forming a diamond-shaped acyclic graph and then pointing to the fifth
//! which is cyclic.
//!
//! ```rust
//! #![feature(rustc_private)]
//!
//! use std::io::Write;
//! use rustc_graphviz as dot;
//!
//! type Nd = isize;
//! type Ed = (isize,isize);
//! struct Edges(Vec<Ed>);
//!
//! pub fn render_to<W: Write>(output: &mut W) {
//!     let edges = Edges(vec![(0,1), (0,2), (1,3), (2,3), (3,4), (4,4)]);
//!     dot::render(&edges, output).unwrap()
//! }
//!
//! impl<'a> dot::Labeller<'a> for Edges {
//!     type Node = Nd;
//!     type Edge = Ed;
//!     fn graph_id(&'a self) -> dot::Id<'a> { dot::Id::new("example1").unwrap() }
//!
//!     fn node_id(&'a self, n: &Nd) -> dot::Id<'a> {
//!         dot::Id::new(format!("N{}", *n)).unwrap()
//!     }
//! }
//!
//! impl<'a> dot::GraphWalk<'a> for Edges {
//!     type Node = Nd;
//!     type Edge = Ed;
//!     fn nodes(&self) -> dot::Nodes<'a,Nd> {
//!         // (assumes that |N| \approxeq |E|)
//!         let &Edges(ref v) = self;
//!         let mut nodes = Vec::with_capacity(v.len());
//!         for &(s,t) in v {
//!             nodes.push(s); nodes.push(t);
//!         }
//!         nodes.sort();
//!         nodes.dedup();
//!         nodes.into()
//!     }
//!
//!     fn edges(&'a self) -> dot::Edges<'a,Ed> {
//!         let &Edges(ref edges) = self;
//!         (&edges[..]).into()
//!     }
//!
//!     fn source(&self, e: &Ed) -> Nd { let &(s,_) = e; s }
//!
//!     fn target(&self, e: &Ed) -> Nd { let &(_,t) = e; t }
//! }
//!
//! # pub fn main() { render_to(&mut Vec::new()) }
//! ```
//!
//! ```no_run
//! # pub fn render_to<W:std::io::Write>(output: &mut W) { unimplemented!() }
//! pub fn main() {
//!     use std::fs::File;
//!     let mut f = File::create("example1.dot").unwrap();
//!     render_to(&mut f)
//! }
//! ```
//!
//! Output from first example (in `example1.dot`):
//!
//! ```dot
//! digraph example1 {
//!     N0[label="N0"];
//!     N1[label="N1"];
//!     N2[label="N2"];
//!     N3[label="N3"];
//!     N4[label="N4"];
//!     N0 -> N1[label=""];
//!     N0 -> N2[label=""];
//!     N1 -> N3[label=""];
//!     N2 -> N3[label=""];
//!     N3 -> N4[label=""];
//!     N4 -> N4[label=""];
//! }
//! ```
//!
//! The second example illustrates using `node_label` and `edge_label` to
//! add labels to the nodes and edges in the rendered graph. The graph
//! here carries both `nodes` (the label text to use for rendering a
//! particular node), and `edges` (again a list of `(source,target)`
//! indices).
//!
//! This example also illustrates how to use a type (in this case the edge
//! type) that shares substructure with the graph: the edge type here is a
//! direct reference to the `(source,target)` pair stored in the graph's
//! internal vector (rather than passing around a copy of the pair
//! itself). Note that this implies that `fn edges(&'a self)` must
//! construct a fresh `Vec<&'a (usize,usize)>` from the `Vec<(usize,usize)>`
//! edges stored in `self`.
//!
//! Since both the set of nodes and the set of edges are always
//! constructed from scratch via iterators, we use the `collect()` method
//! from the `Iterator` trait to collect the nodes and edges into freshly
//! constructed growable `Vec` values (rather than using `Cow` as in the
//! first example above).
//!
//! The output from this example renders four nodes that make up the
//! Hasse-diagram for the subsets of the set `{x, y}`. Each edge is
//! labeled with the &sube; character (specified using the HTML character
//! entity `&sube`).
//!
//! ```rust
//! #![feature(rustc_private)]
//!
//! use std::io::Write;
//! use rustc_graphviz as dot;
//!
//! type Nd = usize;
//! type Ed<'a> = &'a (usize, usize);
//! struct Graph { nodes: Vec<&'static str>, edges: Vec<(usize,usize)> }
//!
//! pub fn render_to<W: Write>(output: &mut W) {
//!     let nodes = vec!["{x,y}","{x}","{y}","{}"];
//!     let edges = vec![(0,1), (0,2), (1,3), (2,3)];
//!     let graph = Graph { nodes: nodes, edges: edges };
//!
//!     dot::render(&graph, output).unwrap()
//! }
//!
//! impl<'a> dot::Labeller<'a> for Graph {
//!     type Node = Nd;
//!     type Edge = Ed<'a>;
//!     fn graph_id(&'a self) -> dot::Id<'a> { dot::Id::new("example2").unwrap() }
//!     fn node_id(&'a self, n: &Nd) -> dot::Id<'a> {
//!         dot::Id::new(format!("N{}", n)).unwrap()
//!     }
//!     fn node_label(&self, n: &Nd) -> dot::LabelText<'_> {
//!         dot::LabelText::LabelStr(self.nodes[*n].into())
//!     }
//!     fn edge_label(&self, _: &Ed<'_>) -> dot::LabelText<'_> {
//!         dot::LabelText::LabelStr("&sube;".into())
//!     }
//! }
//!
//! impl<'a> dot::GraphWalk<'a> for Graph {
//!     type Node = Nd;
//!     type Edge = Ed<'a>;
//!     fn nodes(&self) -> dot::Nodes<'a,Nd> { (0..self.nodes.len()).collect() }
//!     fn edges(&'a self) -> dot::Edges<'a,Ed<'a>> { self.edges.iter().collect() }
//!     fn source(&self, e: &Ed<'_>) -> Nd { let & &(s,_) = e; s }
//!     fn target(&self, e: &Ed<'_>) -> Nd { let & &(_,t) = e; t }
//! }
//!
//! # pub fn main() { render_to(&mut Vec::new()) }
//! ```
//!
//! ```no_run
//! # pub fn render_to<W:std::io::Write>(output: &mut W) { unimplemented!() }
//! pub fn main() {
//!     use std::fs::File;
//!     let mut f = File::create("example2.dot").unwrap();
//!     render_to(&mut f)
//! }
//! ```
//!
//! The third example is similar to the second, except now each node and
//! edge now carries a reference to the string label for each node as well
//! as that node's index. (This is another illustration of how to share
//! structure with the graph itself, and why one might want to do so.)
//!
//! The output from this example is the same as the second example: the
//! Hasse-diagram for the subsets of the set `{x, y}`.
//!
//! ```rust
//! #![feature(rustc_private)]
//!
//! use std::io::Write;
//! use rustc_graphviz as dot;
//!
//! type Nd<'a> = (usize, &'a str);
//! type Ed<'a> = (Nd<'a>, Nd<'a>);
//! struct Graph { nodes: Vec<&'static str>, edges: Vec<(usize,usize)> }
//!
//! pub fn render_to<W: Write>(output: &mut W) {
//!     let nodes = vec!["{x,y}","{x}","{y}","{}"];
//!     let edges = vec![(0,1), (0,2), (1,3), (2,3)];
//!     let graph = Graph { nodes: nodes, edges: edges };
//!
//!     dot::render(&graph, output).unwrap()
//! }
//!
//! impl<'a> dot::Labeller<'a> for Graph {
//!     type Node = Nd<'a>;
//!     type Edge = Ed<'a>;
//!     fn graph_id(&'a self) -> dot::Id<'a> { dot::Id::new("example3").unwrap() }
//!     fn node_id(&'a self, n: &Nd<'a>) -> dot::Id<'a> {
//!         dot::Id::new(format!("N{}", n.0)).unwrap()
//!     }
//!     fn node_label(&self, n: &Nd<'_>) -> dot::LabelText<'_> {
//!         let &(i, _) = n;
//!         dot::LabelText::LabelStr(self.nodes[i].into())
//!     }
//!     fn edge_label(&self, _: &Ed<'_>) -> dot::LabelText<'_> {
//!         dot::LabelText::LabelStr("&sube;".into())
//!     }
//! }
//!
//! impl<'a> dot::GraphWalk<'a> for Graph {
//!     type Node = Nd<'a>;
//!     type Edge = Ed<'a>;
//!     fn nodes(&'a self) -> dot::Nodes<'a,Nd<'a>> {
//!         self.nodes.iter().map(|s| &s[..]).enumerate().collect()
//!     }
//!     fn edges(&'a self) -> dot::Edges<'a,Ed<'a>> {
//!         self.edges.iter()
//!             .map(|&(i,j)|((i, &self.nodes[i][..]),
//!                           (j, &self.nodes[j][..])))
//!             .collect()
//!     }
//!     fn source(&self, e: &Ed<'a>) -> Nd<'a> { let &(s,_) = e; s }
//!     fn target(&self, e: &Ed<'a>) -> Nd<'a> { let &(_,t) = e; t }
//! }
//!
//! # pub fn main() { render_to(&mut Vec::new()) }
//! ```
//!
//! ```no_run
//! # pub fn render_to<W:std::io::Write>(output: &mut W) { unimplemented!() }
//! pub fn main() {
//!     use std::fs::File;
//!     let mut f = File::create("example3.dot").unwrap();
//!     render_to(&mut f)
//! }
//! ```
//!
//! # References
//!
//! * [Graphviz](https://www.graphviz.org/)
//!
//! * [DOT language](https://www.graphviz.org/doc/info/lang.html)

// tidy-alphabetical-start
#![allow(internal_features)]
#![doc(
    html_root_url = "https://doc.rust-lang.org/nightly/nightly-rustc/",
    test(attr(allow(unused_variables), deny(warnings)))
)]
#![doc(rust_logo)]
#![feature(rustdoc_internals)]
// tidy-alphabetical-end

use std::borrow::Cow;
use std::io;
use std::io::prelude::*;

use LabelText::*;

/// The text for a graphviz label on a node or edge.
pub enum LabelText<'a> {
    /// This kind of label preserves the text directly as is.
    ///
    /// Occurrences of backslashes (`\`) are escaped, and thus appear
    /// as backslashes in the rendered label.
    LabelStr(Cow<'a, str>),

    /// This kind of label uses the graphviz label escString type:
    /// <https://www.graphviz.org/docs/attr-types/escString>
    ///
    /// Occurrences of backslashes (`\`) are not escaped; instead they
    /// are interpreted as initiating an escString escape sequence.
    ///
    /// Escape sequences of particular interest: in addition to `\n`
    /// to break a line (centering the line preceding the `\n`), there
    /// are also the escape sequences `\l` which left-justifies the
    /// preceding line and `\r` which right-justifies it.
    EscStr(Cow<'a, str>),

    /// This uses a graphviz [HTML string label][html]. The string is
    /// printed exactly as given, but between `<` and `>`. **No
    /// escaping is performed.**
    ///
    /// [html]: https://www.graphviz.org/doc/info/shapes.html#html
    HtmlStr(Cow<'a, str>),
}

/// The style for a node or edge.
/// See <https://www.graphviz.org/docs/attr-types/style/> for descriptions.
/// Note that some of these are not valid for edges.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum Style {
    None,
    Solid,
    Dashed,
    Dotted,
    Bold,
    Rounded,
    Diagonals,
    Filled,
    Striped,
    Wedged,
}

impl Style {
    pub fn as_slice(self) -> &'static str {
        match self {
            Style::None => "",
            Style::Solid => "solid",
            Style::Dashed => "dashed",
            Style::Dotted => "dotted",
            Style::Bold => "bold",
            Style::Rounded => "rounded",
            Style::Diagonals => "diagonals",
            Style::Filled => "filled",
            Style::Striped => "striped",
            Style::Wedged => "wedged",
        }
    }
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
// structs. See e.g., the `No` and `Ed` structs in the examples. (In
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
    name: Cow<'a, str>,
}

impl<'a> Id<'a> {
    /// Creates an `Id` named `name`.
    ///
    /// The caller must ensure that the input conforms to an
    /// identifier format: it must be a non-empty string made up of
    /// alphanumeric or underscore characters, not beginning with a
    /// digit (i.e., the regular expression `[a-zA-Z_][a-zA-Z_0-9]*`).
    ///
    /// (Note: this format is a strict subset of the `ID` format
    /// defined by the DOT language. This function may change in the
    /// future to accept a broader subset, or the entirety, of DOT's
    /// `ID` format.)
    ///
    /// Passing an invalid string (containing spaces, brackets,
    /// quotes, ...) will return an empty `Err` value.
    pub fn new<Name: Into<Cow<'a, str>>>(name: Name) -> Result<Id<'a>, ()> {
        let name = name.into();
        match name.chars().next() {
            Some(c) if c.is_ascii_alphabetic() || c == '_' => {}
            _ => return Err(()),
        }
        if !name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') {
            return Err(());
        }

        Ok(Id { name })
    }

    pub fn as_slice(&'a self) -> &'a str {
        &self.name
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
pub trait Labeller<'a> {
    type Node;
    type Edge;

    /// Must return a DOT compatible identifier naming the graph.
    fn graph_id(&'a self) -> Id<'a>;

    /// Maps `n` to a unique identifier with respect to `self`. The
    /// implementor is responsible for ensuring that the returned name
    /// is a valid DOT identifier.
    fn node_id(&'a self, n: &Self::Node) -> Id<'a>;

    /// Maps `n` to one of the [graphviz `shape` names][1]. If `None`
    /// is returned, no `shape` attribute is specified.
    ///
    /// [1]: https://www.graphviz.org/doc/info/shapes.html
    fn node_shape(&'a self, _node: &Self::Node) -> Option<LabelText<'a>> {
        None
    }

    /// Maps `n` to a label that will be used in the rendered output.
    /// The label need not be unique, and may be the empty string; the
    /// default is just the output from `node_id`.
    fn node_label(&'a self, n: &Self::Node) -> LabelText<'a> {
        LabelStr(self.node_id(n).name)
    }

    /// Maps `e` to a label that will be used in the rendered output.
    /// The label need not be unique, and may be the empty string; the
    /// default is in fact the empty string.
    fn edge_label(&'a self, _e: &Self::Edge) -> LabelText<'a> {
        LabelStr("".into())
    }

    /// Maps `n` to a style that will be used in the rendered output.
    fn node_style(&'a self, _n: &Self::Node) -> Style {
        Style::None
    }

    /// Maps `e` to a style that will be used in the rendered output.
    fn edge_style(&'a self, _e: &Self::Edge) -> Style {
        Style::None
    }
}

/// Escape tags in such a way that it is suitable for inclusion in a
/// Graphviz HTML label.
pub fn escape_html(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('\"', "&quot;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('\n', "<br align=\"left\"/>")
}

impl<'a> LabelText<'a> {
    pub fn label<S: Into<Cow<'a, str>>>(s: S) -> LabelText<'a> {
        LabelStr(s.into())
    }

    pub fn html<S: Into<Cow<'a, str>>>(s: S) -> LabelText<'a> {
        HtmlStr(s.into())
    }

    fn escape_char<F>(c: char, mut f: F)
    where
        F: FnMut(char),
    {
        match c {
            // not escaping \\, since Graphviz escString needs to
            // interpret backslashes; see EscStr above.
            '\\' => f(c),
            _ => {
                for c in c.escape_default() {
                    f(c)
                }
            }
        }
    }
    fn escape_str(s: &str) -> String {
        let mut out = String::with_capacity(s.len());
        for c in s.chars() {
            LabelText::escape_char(c, |c| out.push(c));
        }
        out
    }

    /// Renders text as string suitable for a label in a .dot file.
    /// This includes quotes or suitable delimiters.
    pub fn to_dot_string(&self) -> String {
        match *self {
            LabelStr(ref s) => format!("\"{}\"", s.escape_default()),
            EscStr(ref s) => format!("\"{}\"", LabelText::escape_str(s)),
            HtmlStr(ref s) => format!("<{s}>"),
        }
    }
}

pub type Nodes<'a, N> = Cow<'a, [N]>;
pub type Edges<'a, E> = Cow<'a, [E]>;

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
/// `Cow<[T]>` to leave implementors the freedom to create
/// entirely new vectors or to pass back slices into internally owned
/// vectors.
pub trait GraphWalk<'a> {
    type Node: Clone;
    type Edge: Clone;

    /// Returns all the nodes in this graph.
    fn nodes(&'a self) -> Nodes<'a, Self::Node>;
    /// Returns all of the edges in this graph.
    fn edges(&'a self) -> Edges<'a, Self::Edge>;
    /// The source node for `edge`.
    fn source(&'a self, edge: &Self::Edge) -> Self::Node;
    /// The target node for `edge`.
    fn target(&'a self, edge: &Self::Edge) -> Self::Node;
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub enum RenderOption {
    NoEdgeLabels,
    NoNodeLabels,
    NoEdgeStyles,
    NoNodeStyles,

    Fontname(String),
    DarkTheme,
}

/// Renders directed graph `g` into the writer `w` in DOT syntax.
/// (Simple wrapper around `render_opts` that passes a default set of options.)
pub fn render<'a, N, E, G, W>(g: &'a G, w: &mut W) -> io::Result<()>
where
    N: Clone + 'a,
    E: Clone + 'a,
    G: Labeller<'a, Node = N, Edge = E> + GraphWalk<'a, Node = N, Edge = E>,
    W: Write,
{
    render_opts(g, w, &[])
}

/// Renders directed graph `g` into the writer `w` in DOT syntax.
/// (Main entry point for the library.)
pub fn render_opts<'a, N, E, G, W>(g: &'a G, w: &mut W, options: &[RenderOption]) -> io::Result<()>
where
    N: Clone + 'a,
    E: Clone + 'a,
    G: Labeller<'a, Node = N, Edge = E> + GraphWalk<'a, Node = N, Edge = E>,
    W: Write,
{
    writeln!(w, "digraph {} {{", g.graph_id().as_slice())?;

    // Global graph properties
    let mut graph_attrs = Vec::new();
    let mut content_attrs = Vec::new();
    let font;
    if let Some(fontname) = options.iter().find_map(|option| {
        if let RenderOption::Fontname(fontname) = option { Some(fontname) } else { None }
    }) {
        font = format!(r#"fontname="{fontname}""#);
        graph_attrs.push(&font[..]);
        content_attrs.push(&font[..]);
    }
    if options.contains(&RenderOption::DarkTheme) {
        graph_attrs.push(r#"bgcolor="black""#);
        graph_attrs.push(r#"fontcolor="white""#);
        content_attrs.push(r#"color="white""#);
        content_attrs.push(r#"fontcolor="white""#);
    }
    if !(graph_attrs.is_empty() && content_attrs.is_empty()) {
        writeln!(w, r#"    graph[{}];"#, graph_attrs.join(" "))?;
        let content_attrs_str = content_attrs.join(" ");
        writeln!(w, r#"    node[{content_attrs_str}];"#)?;
        writeln!(w, r#"    edge[{content_attrs_str}];"#)?;
    }

    let mut text = Vec::new();
    for n in g.nodes().iter() {
        write!(w, "    ")?;
        let id = g.node_id(n);

        let escaped = &g.node_label(n).to_dot_string();

        write!(text, "{}", id.as_slice()).unwrap();

        if !options.contains(&RenderOption::NoNodeLabels) {
            write!(text, "[label={escaped}]").unwrap();
        }

        let style = g.node_style(n);
        if !options.contains(&RenderOption::NoNodeStyles) && style != Style::None {
            write!(text, "[style=\"{}\"]", style.as_slice()).unwrap();
        }

        if let Some(s) = g.node_shape(n) {
            write!(text, "[shape={}]", &s.to_dot_string()).unwrap();
        }

        writeln!(text, ";").unwrap();
        w.write_all(&text)?;

        text.clear();
    }

    for e in g.edges().iter() {
        let escaped_label = &g.edge_label(e).to_dot_string();
        write!(w, "    ")?;
        let source = g.source(e);
        let target = g.target(e);
        let source_id = g.node_id(&source);
        let target_id = g.node_id(&target);

        write!(text, "{} -> {}", source_id.as_slice(), target_id.as_slice()).unwrap();

        if !options.contains(&RenderOption::NoEdgeLabels) {
            write!(text, "[label={escaped_label}]").unwrap();
        }

        let style = g.edge_style(e);
        if !options.contains(&RenderOption::NoEdgeStyles) && style != Style::None {
            write!(text, "[style=\"{}\"]", style.as_slice()).unwrap();
        }

        writeln!(text, ";").unwrap();
        w.write_all(&text)?;

        text.clear();
    }

    writeln!(w, "}}")
}

#[cfg(test)]
mod tests;
