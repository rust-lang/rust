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

use std::borrow::IntoCow;

// For clarity, rename the graphviz crate locally to dot.
use graphviz as dot;

use syntax::ast;
use syntax::ast_map;

use middle::cfg;

pub type Node<'a> = (cfg::CFGIndex, &'a cfg::CFGNode);
pub type Edge<'a> = &'a cfg::CFGEdge;

pub struct LabelledCFG<'a, 'ast: 'a> {
    pub ast_map: &'a ast_map::Map<'ast>,
    pub cfg: &'a cfg::CFG,
    pub name: String,
    /// `labelled_edges` controls whether we emit labels on the edges
    pub labelled_edges: bool,
}

fn replace_newline_with_backslash_l(s: String) -> String {
    // Replacing newlines with \\l causes each line to be left-aligned,
    // improving presentation of (long) pretty-printed expressions.
    if s.contains("\n") {
        let mut s = s.replace("\n", "\\l");
        // Apparently left-alignment applies to the line that precedes
        // \l, not the line that follows; so, add \l at end of string
        // if not already present, ensuring last line gets left-aligned
        // as well.
        let mut last_two: Vec<_> =
            s.chars().rev().take(2).collect();
        last_two.reverse();
        if last_two != ['\\', 'l'] {
            s.push_str("\\l");
        }
        s
    } else {
        s
    }
}

impl<'a, 'ast> dot::Labeller<'a, Node<'a>, Edge<'a>> for LabelledCFG<'a, 'ast> {
    fn graph_id(&'a self) -> dot::Id<'a> { dot::Id::new(&self.name[]).ok().unwrap() }

    fn node_id(&'a self, &(i,_): &Node<'a>) -> dot::Id<'a> {
        dot::Id::new(format!("N{}", i.node_id())).ok().unwrap()
    }

    fn node_label(&'a self, &(i, n): &Node<'a>) -> dot::LabelText<'a> {
        if i == self.cfg.entry {
            dot::LabelText::LabelStr("entry".into_cow())
        } else if i == self.cfg.exit {
            dot::LabelText::LabelStr("exit".into_cow())
        } else if n.data.id == ast::DUMMY_NODE_ID {
            dot::LabelText::LabelStr("(dummy_node)".into_cow())
        } else {
            let s = self.ast_map.node_to_string(n.data.id);
            // left-aligns the lines
            let s = replace_newline_with_backslash_l(s);
            dot::LabelText::EscStr(s.into_cow())
        }
    }

    fn edge_label(&self, e: &Edge<'a>) -> dot::LabelText<'a> {
        let mut label = String::new();
        if !self.labelled_edges {
            return dot::LabelText::EscStr(label.into_cow());
        }
        let mut put_one = false;
        for (i, &node_id) in e.data.exiting_scopes.iter().enumerate() {
            if put_one {
                label.push_str(",\\l");
            } else {
                put_one = true;
            }
            let s = self.ast_map.node_to_string(node_id);
            // left-aligns the lines
            let s = replace_newline_with_backslash_l(s);
            label.push_str(&format!("exiting scope_{} {}",
                                   i,
                                   &s[..])[]);
        }
        dot::LabelText::EscStr(label.into_cow())
    }
}

impl<'a> dot::GraphWalk<'a, Node<'a>, Edge<'a>> for &'a cfg::CFG {
    fn nodes(&'a self) -> dot::Nodes<'a, Node<'a>> {
        let mut v = Vec::new();
        self.graph.each_node(|i, nd| { v.push((i, nd)); true });
        v.into_cow()
    }
    fn edges(&'a self) -> dot::Edges<'a, Edge<'a>> {
        self.graph.all_edges().iter().collect()
    }
    fn source(&'a self, edge: &Edge<'a>) -> Node<'a> {
        let i = edge.source();
        (i, self.graph.node(i))
    }
    fn target(&'a self, edge: &Edge<'a>) -> Node<'a> {
        let i = edge.target();
        (i, self.graph.node(i))
    }
}

impl<'a, 'ast> dot::GraphWalk<'a, Node<'a>, Edge<'a>> for LabelledCFG<'a, 'ast>
{
    fn nodes(&'a self) -> dot::Nodes<'a, Node<'a>> { self.cfg.nodes() }
    fn edges(&'a self) -> dot::Edges<'a, Edge<'a>> { self.cfg.edges() }
    fn source(&'a self, edge: &Edge<'a>) -> Node<'a> { self.cfg.source(edge) }
    fn target(&'a self, edge: &Edge<'a>) -> Node<'a> { self.cfg.target(edge) }
}

