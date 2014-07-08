// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/// This module provides linkage between rustc::middle::graph and
/// libgraphviz traits.

/// For clarity, rename the graphviz crate locally to dot.
use dot = graphviz;

use syntax::ast;
use syntax::ast_map;

use middle::cfg;

pub type Node<'a> = (cfg::CFGIndex, &'a cfg::CFGNode);
pub type Edge<'a> = &'a cfg::CFGEdge;

pub struct LabelledCFG<'a>{
    pub ast_map: &'a ast_map::Map,
    pub cfg: &'a cfg::CFG,
    pub name: String,
}

fn replace_newline_with_backslash_l(s: String) -> String {
    // Replacing newlines with \\l causes each line to be left-aligned,
    // improving presentation of (long) pretty-printed expressions.
    if s.as_slice().contains("\n") {
        let mut s = s.replace("\n", "\\l");
        // Apparently left-alignment applies to the line that precedes
        // \l, not the line that follows; so, add \l at end of string
        // if not already present, ensuring last line gets left-aligned
        // as well.
        let mut last_two: Vec<_> =
            s.as_slice().chars().rev().take(2).collect();
        last_two.reverse();
        if last_two.as_slice() != ['\\', 'l'] {
            s = s.append("\\l");
        }
        s.to_string()
    } else {
        s
    }
}

impl<'a> dot::Labeller<'a, Node<'a>, Edge<'a>> for LabelledCFG<'a> {
    fn graph_id(&'a self) -> dot::Id<'a> { dot::Id::new(self.name.as_slice()) }

    fn node_id(&'a self, &(i,_): &Node<'a>) -> dot::Id<'a> {
        dot::Id::new(format!("N{:u}", i.node_id()))
    }

    fn node_label(&'a self, &(i, n): &Node<'a>) -> dot::LabelText<'a> {
        if i == self.cfg.entry {
            dot::LabelStr("entry".into_maybe_owned())
        } else if i == self.cfg.exit {
            dot::LabelStr("exit".into_maybe_owned())
        } else if n.data.id == ast::DUMMY_NODE_ID {
            dot::LabelStr("(dummy_node)".into_maybe_owned())
        } else {
            let s = self.ast_map.node_to_string(n.data.id);
            // left-aligns the lines
            let s = replace_newline_with_backslash_l(s);
            dot::EscStr(s.into_maybe_owned())
        }
    }

    fn edge_label(&self, e: &Edge<'a>) -> dot::LabelText<'a> {
        let mut label = String::new();
        let mut put_one = false;
        for (i, &node_id) in e.data.exiting_scopes.iter().enumerate() {
            if put_one {
                label = label.append(",\\l");
            } else {
                put_one = true;
            }
            let s = self.ast_map.node_to_string(node_id);
            // left-aligns the lines
            let s = replace_newline_with_backslash_l(s);
            label = label.append(format!("exiting scope_{} {}",
                                         i,
                                         s.as_slice()).as_slice());
        }
        dot::EscStr(label.into_maybe_owned())
    }
}

impl<'a> dot::GraphWalk<'a, Node<'a>, Edge<'a>> for &'a cfg::CFG {
    fn nodes(&self) -> dot::Nodes<'a, Node<'a>> {
        let mut v = Vec::new();
        self.graph.each_node(|i, nd| { v.push((i, nd)); true });
        dot::maybe_owned_vec::Growable(v)
    }
    fn edges(&self) -> dot::Edges<'a, Edge<'a>> {
        self.graph.all_edges().iter().collect()
    }
    fn source(&self, edge: &Edge<'a>) -> Node<'a> {
        let i = edge.source();
        (i, self.graph.node(i))
    }
    fn target(&self, edge: &Edge<'a>) -> Node<'a> {
        let i = edge.target();
        (i, self.graph.node(i))
    }
}

impl<'a> dot::GraphWalk<'a, Node<'a>, Edge<'a>> for LabelledCFG<'a>
{
    fn nodes(&self) -> dot::Nodes<'a, Node<'a>> { self.cfg.nodes() }
    fn edges(&self) -> dot::Edges<'a, Edge<'a>> { self.cfg.edges() }
    fn source(&self, edge: &Edge<'a>) -> Node<'a> { self.cfg.source(edge) }
    fn target(&self, edge: &Edge<'a>) -> Node<'a> { self.cfg.target(edge) }
}
