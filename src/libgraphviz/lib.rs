// Copyright 2014-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Generate files suitable for use with [Graphviz](http://www.graphviz.org/)
//!
//! The `render` function generates output (e.g. an `output.dot` file) for
//! use with [Graphviz](http://www.graphviz.org/) by walking a labelled
//! graph. (Graphviz can then automatically lay out the nodes and edges
//! of the graph, and also optionally render the graph as an image or
//! other [output formats](
//! http://www.graphviz.org/content/output-formats), such as SVG.)
//!
//! Rather than impose some particular graph data structure on clients,
//! this library exposes two traits that clients can implement on their
//! own structs before handing them over to the rendering function.
//!
//! Note: This library does not yet provide access to the full
//! expressiveness of the [DOT language](
//! http://www.graphviz.org/doc/info/lang.html). For example, there are
//! many [attributes](http://www.graphviz.org/content/attrs) related to
//! providing layout hints (e.g. left-to-right versus top-down, which
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
//! #![feature(rustc_private, into_cow)]
//!
//! use std::borrow::IntoCow;
//! use std::io::Write;
//! use graphviz as dot;
//!
//! type Nd = isize;
//! type Ed = (isize,isize);
//! struct Edges(Vec<Ed>);
//!
//! pub fn render_to<W: Write>(output: &mut W) {
//!     let edges = Edges(vec!((0,1), (0,2), (1,3), (2,3), (3,4), (4,4)));
//!     dot::render(&edges, output).unwrap()
//! }
//!
//! impl<'a> dot::Labeller<'a, Nd, Ed> for Edges {
//!     fn graph_id(&'a self) -> dot::Id<'a> { dot::Id::new("example1").unwrap() }
//!
//!     fn node_id(&'a self, n: &Nd) -> dot::Id<'a> {
//!         dot::Id::new(format!("N{}", *n)).unwrap()
//!     }
//! }
//!
//! impl<'a> dot::GraphWalk<'a, Nd, Ed> for Edges {
//!     fn nodes(&self) -> dot::Nodes<'a,Nd> {
//!         // (assumes that |N| \approxeq |E|)
//!         let &Edges(ref v) = self;
//!         let mut nodes = Vec::with_capacity(v.len());
//!         for &(s,t) in v {
//!             nodes.push(s); nodes.push(t);
//!         }
//!         nodes.sort();
//!         nodes.dedup();
//!         nodes.into_cow()
//!     }
//!
//!     fn edges(&'a self) -> dot::Edges<'a,Ed> {
//!         let &Edges(ref edges) = self;
//!         (&edges[..]).into_cow()
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
//! ```ignore
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
//! constructed growable `Vec` values (rather use the `into_cow`
//! from the `IntoCow` trait as was used in the first example
//! above).
//!
//! The output from this example renders four nodes that make up the
//! Hasse-diagram for the subsets of the set `{x, y}`. Each edge is
//! labelled with the &sube; character (specified using the HTML character
//! entity `&sube`).
//!
//! ```rust
//! #![feature(rustc_private)]
//!
//! use std::io::Write;
//! use graphviz as dot;
//!
//! type Nd = usize;
//! type Ed<'a> = &'a (usize, usize);
//! struct Graph { nodes: Vec<&'static str>, edges: Vec<(usize,usize)> }
//!
//! pub fn render_to<W: Write>(output: &mut W) {
//!     let nodes = vec!("{x,y}","{x}","{y}","{}");
//!     let edges = vec!((0,1), (0,2), (1,3), (2,3));
//!     let graph = Graph { nodes: nodes, edges: edges };
//!
//!     dot::render(&graph, output).unwrap()
//! }
//!
//! impl<'a> dot::Labeller<'a, Nd, Ed<'a>> for Graph {
//!     fn graph_id(&'a self) -> dot::Id<'a> { dot::Id::new("example2").unwrap() }
//!     fn node_id(&'a self, n: &Nd) -> dot::Id<'a> {
//!         dot::Id::new(format!("N{}", n)).unwrap()
//!     }
//!     fn node_label<'b>(&'b self, n: &Nd) -> dot::LabelText<'b> {
//!         dot::LabelText::LabelStr(self.nodes[*n].into())
//!     }
//!     fn edge_label<'b>(&'b self, _: &Ed) -> dot::LabelText<'b> {
//!         dot::LabelText::LabelStr("&sube;".into())
//!     }
//! }
//!
//! impl<'a> dot::GraphWalk<'a, Nd, Ed<'a>> for Graph {
//!     fn nodes(&self) -> dot::Nodes<'a,Nd> { (0..self.nodes.len()).collect() }
//!     fn edges(&'a self) -> dot::Edges<'a,Ed<'a>> { self.edges.iter().collect() }
//!     fn source(&self, e: &Ed) -> Nd { let & &(s,_) = e; s }
//!     fn target(&self, e: &Ed) -> Nd { let & &(_,t) = e; t }
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
//! use graphviz as dot;
//!
//! type Nd<'a> = (usize, &'a str);
//! type Ed<'a> = (Nd<'a>, Nd<'a>);
//! struct Graph { nodes: Vec<&'static str>, edges: Vec<(usize,usize)> }
//!
//! pub fn render_to<W: Write>(output: &mut W) {
//!     let nodes = vec!("{x,y}","{x}","{y}","{}");
//!     let edges = vec!((0,1), (0,2), (1,3), (2,3));
//!     let graph = Graph { nodes: nodes, edges: edges };
//!
//!     dot::render(&graph, output).unwrap()
//! }
//!
//! impl<'a> dot::Labeller<'a, Nd<'a>, Ed<'a>> for Graph {
//!     fn graph_id(&'a self) -> dot::Id<'a> { dot::Id::new("example3").unwrap() }
//!     fn node_id(&'a self, n: &Nd<'a>) -> dot::Id<'a> {
//!         dot::Id::new(format!("N{}", n.0)).unwrap()
//!     }
//!     fn node_label<'b>(&'b self, n: &Nd<'b>) -> dot::LabelText<'b> {
//!         let &(i, _) = n;
//!         dot::LabelText::LabelStr(self.nodes[i].into())
//!     }
//!     fn edge_label<'b>(&'b self, _: &Ed<'b>) -> dot::LabelText<'b> {
//!         dot::LabelText::LabelStr("&sube;".into())
//!     }
//! }
//!
//! impl<'a> dot::GraphWalk<'a, Nd<'a>, Ed<'a>> for Graph {
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
//! * [Graphviz](http://www.graphviz.org/)
//!
//! * [DOT language](http://www.graphviz.org/doc/info/lang.html)

// Do not remove on snapshot creation. Needed for bootstrap. (Issue #22364)
#![cfg_attr(stage0, feature(custom_attribute))]
#![crate_name = "graphviz"]
#![unstable(feature = "rustc_private", issue = "27812")]
#![feature(staged_api)]
#![cfg_attr(stage0, staged_api)]
#![crate_type = "rlib"]
#![crate_type = "dylib"]
#![doc(html_logo_url = "https://www.rust-lang.org/logos/rust-logo-128x128-blk-v2.png",
       html_favicon_url = "https://doc.rust-lang.org/favicon.ico",
       html_root_url = "https://doc.rust-lang.org/nightly/",
       test(attr(allow(unused_variables), deny(warnings))))]

#![feature(into_cow)]
#![feature(str_escape)]

use self::LabelText::*;

use std::borrow::{IntoCow, Cow};
use std::io::prelude::*;
use std::io;

/// The text for a graphviz label on a node or edge.
pub enum LabelText<'a> {
    /// This kind of label preserves the text directly as is.
    ///
    /// Occurrences of backslashes (`\`) are escaped, and thus appear
    /// as backslashes in the rendered label.
    LabelStr(Cow<'a, str>),

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
    EscStr(Cow<'a, str>),

    /// This uses a graphviz [HTML string label][html]. The string is
    /// printed exactly as given, but between `<` and `>`. **No
    /// escaping is performed.**
    ///
    /// [html]: http://www.graphviz.org/content/node-shapes#html
    HtmlStr(Cow<'a, str>),
}

/// The style for a node or edge.
/// See http://www.graphviz.org/doc/info/attrs.html#k:style for descriptions.
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
    name: Cow<'a, str>,
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
    ///
    /// Passing an invalid string (containing spaces, brackets,
    /// quotes, ...) will return an empty `Err` value.
    pub fn new<Name: IntoCow<'a, str>>(name: Name) -> Result<Id<'a>, ()> {
        let name = name.into_cow();
        {
            let mut chars = name.chars();
            match chars.next() {
                Some(c) if is_letter_or_underscore(c) => {}
                _ => return Err(()),
            }
            if !chars.all(is_constituent) {
                return Err(())
            }
        }
        return Ok(Id{ name: name });

        fn is_letter_or_underscore(c: char) -> bool {
            in_range('a', c, 'z') || in_range('A', c, 'Z') || c == '_'
        }
        fn is_constituent(c: char) -> bool {
            is_letter_or_underscore(c) || in_range('0', c, '9')
        }
        fn in_range(low: char, c: char, high: char) -> bool {
            low as usize <= c as usize && c as usize <= high as usize
        }
    }

    pub fn as_slice(&'a self) -> &'a str {
        &*self.name
    }

    pub fn name(self) -> Cow<'a, str> {
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
    /// implementor is responsible for ensuring that the returned name
    /// is a valid DOT identifier.
    fn node_id(&'a self, n: &N) -> Id<'a>;

    /// Maps `n` to one of the [graphviz `shape` names][1]. If `None`
    /// is returned, no `shape` attribute is specified.
    ///
    /// [1]: http://www.graphviz.org/content/node-shapes
    fn node_shape(&'a self, _node: &N) -> Option<LabelText<'a>> {
        None
    }

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
        LabelStr("".into_cow())
    }

    /// Maps `n` to a style that will be used in the rendered output.
    fn node_style(&'a self, _n: &N) -> Style {
        Style::None
    }

    /// Maps `e` to a style that will be used in the rendered output.
    fn edge_style(&'a self, _e: &E) -> Style {
        Style::None
    }
}

/// Escape tags in such a way that it is suitable for inclusion in a
/// Graphviz HTML label.
pub fn escape_html(s: &str) -> String {
    s
        .replace("&", "&amp;")
        .replace("\"", "&quot;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
}

impl<'a> LabelText<'a> {
    pub fn label<S: IntoCow<'a, str>>(s: S) -> LabelText<'a> {
        LabelStr(s.into_cow())
    }

    pub fn escaped<S: IntoCow<'a, str>>(s: S) -> LabelText<'a> {
        EscStr(s.into_cow())
    }

    pub fn html<S: IntoCow<'a, str>>(s: S) -> LabelText<'a> {
        HtmlStr(s.into_cow())
    }

    fn escape_char<F>(c: char, mut f: F)
        where F: FnMut(char)
    {
        match c {
            // not escaping \\, since Graphviz escString needs to
            // interpret backslashes; see EscStr above.
            '\\' => f(c),
            _ => for c in c.escape_default() {
                f(c)
            },
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
    /// This includes quotes or suitable delimeters.
    pub fn to_dot_string(&self) -> String {
        match self {
            &LabelStr(ref s) => format!("\"{}\"", s.escape_default()),
            &EscStr(ref s) => format!("\"{}\"", LabelText::escape_str(&s[..])),
            &HtmlStr(ref s) => format!("<{}>", s),
        }
    }

    /// Decomposes content into string suitable for making EscStr that
    /// yields same content as self.  The result obeys the law
    /// render(`lt`) == render(`EscStr(lt.pre_escaped_content())`) for
    /// all `lt: LabelText`.
    fn pre_escaped_content(self) -> Cow<'a, str> {
        match self {
            EscStr(s) => s,
            LabelStr(s) => if s.contains('\\') {
                (&*s).escape_default().into_cow()
            } else {
                s
            },
            HtmlStr(s) => s,
        }
    }

    /// Puts `prefix` on a line above this label, with a blank line separator.
    pub fn prefix_line(self, prefix: LabelText) -> LabelText<'static> {
        prefix.suffix_line(self)
    }

    /// Puts `suffix` on a line below this label, with a blank line separator.
    pub fn suffix_line(self, suffix: LabelText) -> LabelText<'static> {
        let mut prefix = self.pre_escaped_content().into_owned();
        let suffix = suffix.pre_escaped_content();
        prefix.push_str(r"\n\n");
        prefix.push_str(&suffix[..]);
        EscStr(prefix.into_cow())
    }
}

pub type Nodes<'a,N> = Cow<'a,[N]>;
pub type Edges<'a,E> = Cow<'a,[E]>;

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
pub trait GraphWalk<'a, N: Clone, E: Clone> {
    /// Returns all the nodes in this graph.
    fn nodes(&'a self) -> Nodes<'a, N>;
    /// Returns all of the edges in this graph.
    fn edges(&'a self) -> Edges<'a, E>;
    /// The source node for `edge`.
    fn source(&'a self, edge: &E) -> N;
    /// The target node for `edge`.
    fn target(&'a self, edge: &E) -> N;
}

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum RenderOption {
    NoEdgeLabels,
    NoNodeLabels,
    NoEdgeStyles,
    NoNodeStyles,
}

/// Returns vec holding all the default render options.
pub fn default_options() -> Vec<RenderOption> {
    vec![]
}

/// Renders directed graph `g` into the writer `w` in DOT syntax.
/// (Simple wrapper around `render_opts` that passes a default set of options.)
pub fn render<'a,
              N: Clone + 'a,
              E: Clone + 'a,
              G: Labeller<'a, N, E> + GraphWalk<'a, N, E>,
              W: Write>
    (g: &'a G,
     w: &mut W)
     -> io::Result<()> {
    render_opts(g, w, &[])
}

/// Renders directed graph `g` into the writer `w` in DOT syntax.
/// (Main entry point for the library.)
pub fn render_opts<'a,
                   N: Clone + 'a,
                   E: Clone + 'a,
                   G: Labeller<'a, N, E> + GraphWalk<'a, N, E>,
                   W: Write>
    (g: &'a G,
     w: &mut W,
     options: &[RenderOption])
     -> io::Result<()> {
    fn writeln<W: Write>(w: &mut W, arg: &[&str]) -> io::Result<()> {
        for &s in arg {
            try!(w.write_all(s.as_bytes()));
        }
        write!(w, "\n")
    }

    fn indent<W: Write>(w: &mut W) -> io::Result<()> {
        w.write_all(b"    ")
    }

    try!(writeln(w, &["digraph ", g.graph_id().as_slice(), " {"]));
    for n in g.nodes().iter() {
        try!(indent(w));
        let id = g.node_id(n);

        let escaped = &g.node_label(n).to_dot_string();
        let shape;

        let mut text = vec![id.as_slice()];

        if !options.contains(&RenderOption::NoNodeLabels) {
            text.push("[label=");
            text.push(escaped);
            text.push("]");
        }

        let style = g.node_style(n);
        if !options.contains(&RenderOption::NoNodeStyles) && style != Style::None {
            text.push("[style=\"");
            text.push(style.as_slice());
            text.push("\"]");
        }

        if let Some(s) = g.node_shape(n) {
            shape = s.to_dot_string();
            text.push("[shape=");
            text.push(&shape);
            text.push("]");
        }

        text.push(";");
        try!(writeln(w, &text));
    }

    for e in g.edges().iter() {
        let escaped_label = &g.edge_label(e).to_dot_string();
        try!(indent(w));
        let source = g.source(e);
        let target = g.target(e);
        let source_id = g.node_id(&source);
        let target_id = g.node_id(&target);

        let mut text = vec![source_id.as_slice(), " -> ", target_id.as_slice()];

        if !options.contains(&RenderOption::NoEdgeLabels) {
            text.push("[label=");
            text.push(escaped_label);
            text.push("]");
        }

        let style = g.edge_style(e);
        if !options.contains(&RenderOption::NoEdgeStyles) && style != Style::None {
            text.push("[style=\"");
            text.push(style.as_slice());
            text.push("\"]");
        }

        text.push(";");
        try!(writeln(w, &text));
    }

    writeln(w, &["}"])
}

#[cfg(test)]
mod tests {
    use self::NodeLabels::*;
    use super::{Id, Labeller, Nodes, Edges, GraphWalk, render, Style};
    use super::LabelText::{self, LabelStr, EscStr, HtmlStr};
    use std::io;
    use std::io::prelude::*;
    use std::borrow::IntoCow;

    /// each node is an index in a vector in the graph.
    type Node = usize;
    struct Edge {
        from: usize,
        to: usize,
        label: &'static str,
        style: Style,
    }

    fn edge(from: usize, to: usize, label: &'static str, style: Style) -> Edge {
        Edge { from: from, to: to, label: label, style: style }
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
                AllNodesLabelled(lbls) => lbls.into_iter().map(|l| Some(l)).collect(),
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
        fn new(name: &'static str,
               node_labels: Trivial,
               edges: Vec<Edge>,
               node_styles: Option<Vec<Style>>)
               -> LabelledGraph {
            let count = node_labels.len();
            LabelledGraph {
                name: name,
                node_labels: node_labels.to_opt_strs(),
                edges: edges,
                node_styles: match node_styles {
                    Some(nodes) => nodes,
                    None => vec![Style::None; count],
                },
            }
        }
    }

    impl LabelledGraphWithEscStrs {
        fn new(name: &'static str,
               node_labels: Trivial,
               edges: Vec<Edge>)
               -> LabelledGraphWithEscStrs {
            LabelledGraphWithEscStrs { graph: LabelledGraph::new(name, node_labels, edges, None) }
        }
    }

    fn id_name<'a>(n: &Node) -> Id<'a> {
        Id::new(format!("N{}", *n)).unwrap()
    }

    impl<'a> Labeller<'a, Node, &'a Edge> for LabelledGraph {
        fn graph_id(&'a self) -> Id<'a> {
            Id::new(&self.name[..]).unwrap()
        }
        fn node_id(&'a self, n: &Node) -> Id<'a> {
            id_name(n)
        }
        fn node_label(&'a self, n: &Node) -> LabelText<'a> {
            match self.node_labels[*n] {
                Some(ref l) => LabelStr(l.into_cow()),
                None => LabelStr(id_name(n).name()),
            }
        }
        fn edge_label(&'a self, e: &&'a Edge) -> LabelText<'a> {
            LabelStr(e.label.into_cow())
        }
        fn node_style(&'a self, n: &Node) -> Style {
            self.node_styles[*n]
        }
        fn edge_style(&'a self, e: &&'a Edge) -> Style {
            e.style
        }
    }

    impl<'a> Labeller<'a, Node, &'a Edge> for LabelledGraphWithEscStrs {
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

    impl<'a> GraphWalk<'a, Node, &'a Edge> for LabelledGraph {
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

    impl<'a> GraphWalk<'a, Node, &'a Edge> for LabelledGraphWithEscStrs {
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
        try!(Read::read_to_string(&mut &*writer, &mut s));
        Ok(s)
    }

    // All of the tests use raw-strings as the format for the expected outputs,
    // so that you can cut-and-paste the content into a .dot file yourself to
    // see what the graphviz visualizer would produce.

    #[test]
    fn empty_graph() {
        let labels: Trivial = UnlabelledNodes(0);
        let r = test_input(LabelledGraph::new("empty_graph", labels, vec![], None));
        assert_eq!(r.unwrap(),
r#"digraph empty_graph {
}
"#);
    }

    #[test]
    fn single_node() {
        let labels: Trivial = UnlabelledNodes(1);
        let r = test_input(LabelledGraph::new("single_node", labels, vec![], None));
        assert_eq!(r.unwrap(),
r#"digraph single_node {
    N0[label="N0"];
}
"#);
    }

    #[test]
    fn single_node_with_style() {
        let labels: Trivial = UnlabelledNodes(1);
        let styles = Some(vec![Style::Dashed]);
        let r = test_input(LabelledGraph::new("single_node", labels, vec![], styles));
        assert_eq!(r.unwrap(),
r#"digraph single_node {
    N0[label="N0"][style="dashed"];
}
"#);
    }

    #[test]
    fn single_edge() {
        let labels: Trivial = UnlabelledNodes(2);
        let result = test_input(LabelledGraph::new("single_edge",
                                                   labels,
                                                   vec![edge(0, 1, "E", Style::None)],
                                                   None));
        assert_eq!(result.unwrap(),
r#"digraph single_edge {
    N0[label="N0"];
    N1[label="N1"];
    N0 -> N1[label="E"];
}
"#);
    }

    #[test]
    fn single_edge_with_style() {
        let labels: Trivial = UnlabelledNodes(2);
        let result = test_input(LabelledGraph::new("single_edge",
                                                   labels,
                                                   vec![edge(0, 1, "E", Style::Bold)],
                                                   None));
        assert_eq!(result.unwrap(),
r#"digraph single_edge {
    N0[label="N0"];
    N1[label="N1"];
    N0 -> N1[label="E"][style="bold"];
}
"#);
    }

    #[test]
    fn test_some_labelled() {
        let labels: Trivial = SomeNodesLabelled(vec![Some("A"), None]);
        let styles = Some(vec![Style::None, Style::Dotted]);
        let result = test_input(LabelledGraph::new("test_some_labelled",
                                                   labels,
                                                   vec![edge(0, 1, "A-1", Style::None)],
                                                   styles));
        assert_eq!(result.unwrap(),
r#"digraph test_some_labelled {
    N0[label="A"];
    N1[label="N1"][style="dotted"];
    N0 -> N1[label="A-1"];
}
"#);
    }

    #[test]
    fn single_cyclic_node() {
        let labels: Trivial = UnlabelledNodes(1);
        let r = test_input(LabelledGraph::new("single_cyclic_node",
                                              labels,
                                              vec![edge(0, 0, "E", Style::None)],
                                              None));
        assert_eq!(r.unwrap(),
r#"digraph single_cyclic_node {
    N0[label="N0"];
    N0 -> N0[label="E"];
}
"#);
    }

    #[test]
    fn hasse_diagram() {
        let labels = AllNodesLabelled(vec!("{x,y}", "{x}", "{y}", "{}"));
        let r = test_input(LabelledGraph::new("hasse_diagram",
                                              labels,
                                              vec![edge(0, 1, "", Style::None),
                                                   edge(0, 2, "", Style::None),
                                                   edge(1, 3, "", Style::None),
                                                   edge(2, 3, "", Style::None)],
                                              None));
        assert_eq!(r.unwrap(),
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

        let mut writer = Vec::new();

        let g = LabelledGraphWithEscStrs::new("syntax_tree",
                                              labels,
                                              vec![edge(0, 1, "then", Style::None),
                                                   edge(0, 2, "else", Style::None),
                                                   edge(1, 3, ";", Style::None),
                                                   edge(2, 3, ";", Style::None)]);

        render(&g, &mut writer).unwrap();
        let mut r = String::new();
        Read::read_to_string(&mut &*writer, &mut r).unwrap();

        assert_eq!(r,
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
}
