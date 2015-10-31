// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! This module provides linkage between libgraphviz traits and
//! `rustc::middle::typeck::infer::region_inference`, generating a
//! rendering of the graph represented by the list of `Constraint`
//! instances (which make up the edges of the graph), as well as the
//! origin for each constraint (which are attached to the labels on
//! each edge).

/// For clarity, rename the graphviz crate locally to dot.
use graphviz as dot;

use middle::ty;
use middle::region::CodeExtent;
use super::Constraint;
use middle::infer::SubregionOrigin;
use middle::infer::region_inference::RegionVarBindings;
use util::nodemap::{FnvHashMap, FnvHashSet};

use std::borrow::Cow;
use std::collections::hash_map::Entry::Vacant;
use std::env;
use std::fs::File;
use std::io;
use std::io::prelude::*;
use std::sync::atomic::{AtomicBool, Ordering};
use syntax::ast;

fn print_help_message() {
    println!("\
-Z print-region-graph by default prints a region constraint graph for every \n\
function body, to the path `/tmp/constraints.nodeXXX.dot`, where the XXX is \n\
replaced with the node id of the function under analysis.                   \n\
                                                                            \n\
To select one particular function body, set `RUST_REGION_GRAPH_NODE=XXX`,   \n\
where XXX is the node id desired.                                           \n\
                                                                            \n\
To generate output to some path other than the default                      \n\
`/tmp/constraints.nodeXXX.dot`, set `RUST_REGION_GRAPH=/path/desired.dot`;  \n\
occurrences of the character `%` in the requested path will be replaced with\n\
the node id of the function under analysis.                                 \n\
                                                                            \n\
(Since you requested help via RUST_REGION_GRAPH=help, no region constraint  \n\
graphs will be printed.                                                     \n\
");
}

pub fn maybe_print_constraints_for<'a, 'tcx>(region_vars: &RegionVarBindings<'a, 'tcx>,
                                             subject_node: ast::NodeId) {
    let tcx = region_vars.tcx;

    if !region_vars.tcx.sess.opts.debugging_opts.print_region_graph {
        return;
    }

    let requested_node: Option<ast::NodeId> = env::var("RUST_REGION_GRAPH_NODE")
                                                  .ok()
                                                  .and_then(|s| s.parse().ok());

    if requested_node.is_some() && requested_node != Some(subject_node) {
        return;
    }

    let requested_output = env::var("RUST_REGION_GRAPH");
    debug!("requested_output: {:?} requested_node: {:?}",
           requested_output,
           requested_node);

    let output_path = {
        let output_template = match requested_output {
            Ok(ref s) if &**s == "help" => {
                static PRINTED_YET: AtomicBool = AtomicBool::new(false);
                if !PRINTED_YET.load(Ordering::SeqCst) {
                    print_help_message();
                    PRINTED_YET.store(true, Ordering::SeqCst);
                }
                return;
            }

            Ok(other_path) => other_path,
            Err(_) => "/tmp/constraints.node%.dot".to_string(),
        };

        if output_template.is_empty() {
            tcx.sess.bug("empty string provided as RUST_REGION_GRAPH");
        }

        if output_template.contains('%') {
            let mut new_str = String::new();
            for c in output_template.chars() {
                if c == '%' {
                    new_str.push_str(&subject_node.to_string());
                } else {
                    new_str.push(c);
                }
            }
            new_str
        } else {
            output_template
        }
    };

    let constraints = &*region_vars.constraints.borrow();
    match dump_region_constraints_to(tcx, constraints, &output_path) {
        Ok(()) => {}
        Err(e) => {
            let msg = format!("io error dumping region constraints: {}", e);
            region_vars.tcx.sess.err(&msg)
        }
    }
}

struct ConstraintGraph<'a, 'tcx: 'a> {
    tcx: &'a ty::ctxt<'tcx>,
    graph_name: String,
    map: &'a FnvHashMap<Constraint, SubregionOrigin<'tcx>>,
    node_ids: FnvHashMap<Node, usize>,
}

#[derive(Clone, Hash, PartialEq, Eq, Debug, Copy)]
enum Node {
    RegionVid(ty::RegionVid),
    Region(ty::Region),
}

// type Edge = Constraint;
#[derive(Clone, PartialEq, Eq, Debug, Copy)]
enum Edge {
    Constraint(Constraint),
    EnclScope(CodeExtent, CodeExtent),
}

impl<'a, 'tcx> ConstraintGraph<'a, 'tcx> {
    fn new(tcx: &'a ty::ctxt<'tcx>,
           name: String,
           map: &'a ConstraintMap<'tcx>)
           -> ConstraintGraph<'a, 'tcx> {
        let mut i = 0;
        let mut node_ids = FnvHashMap();
        {
            let mut add_node = |node| {
                if let Vacant(e) = node_ids.entry(node) {
                    e.insert(i);
                    i += 1;
                }
            };

            for (n1, n2) in map.keys().map(|c| constraint_to_nodes(c)) {
                add_node(n1);
                add_node(n2);
            }

            tcx.region_maps.each_encl_scope(|sub, sup| {
                add_node(Node::Region(ty::ReScope(*sub)));
                add_node(Node::Region(ty::ReScope(*sup)));
            });
        }

        ConstraintGraph {
            tcx: tcx,
            graph_name: name,
            map: map,
            node_ids: node_ids,
        }
    }
}

impl<'a, 'tcx> dot::Labeller<'a, Node, Edge> for ConstraintGraph<'a, 'tcx> {
    fn graph_id(&self) -> dot::Id {
        dot::Id::new(&*self.graph_name).unwrap()
    }
    fn node_id(&self, n: &Node) -> dot::Id {
        let node_id = match self.node_ids.get(n) {
            Some(node_id) => node_id,
            None => panic!("no node_id found for node: {:?}", n),
        };
        let name = || format!("node_{}", node_id);
        match dot::Id::new(name()) {
            Ok(id) => id,
            Err(_) => {
                panic!("failed to create graphviz node identified by {}", name());
            }
        }
    }
    fn node_label(&self, n: &Node) -> dot::LabelText {
        match *n {
            Node::RegionVid(n_vid) => dot::LabelText::label(format!("{:?}", n_vid)),
            Node::Region(n_rgn) => dot::LabelText::label(format!("{:?}", n_rgn)),
        }
    }
    fn edge_label(&self, e: &Edge) -> dot::LabelText {
        match *e {
            Edge::Constraint(ref c) =>
                dot::LabelText::label(format!("{:?}", self.map.get(c).unwrap())),
            Edge::EnclScope(..) => dot::LabelText::label(format!("(enclosed)")),
        }
    }
}

fn constraint_to_nodes(c: &Constraint) -> (Node, Node) {
    match *c {
        Constraint::ConstrainVarSubVar(rv_1, rv_2) =>
            (Node::RegionVid(rv_1), Node::RegionVid(rv_2)),
        Constraint::ConstrainRegSubVar(r_1, rv_2) => (Node::Region(r_1), Node::RegionVid(rv_2)),
        Constraint::ConstrainVarSubReg(rv_1, r_2) => (Node::RegionVid(rv_1), Node::Region(r_2)),
    }
}

fn edge_to_nodes(e: &Edge) -> (Node, Node) {
    match *e {
        Edge::Constraint(ref c) => constraint_to_nodes(c),
        Edge::EnclScope(sub, sup) => {
            (Node::Region(ty::ReScope(sub)),
             Node::Region(ty::ReScope(sup)))
        }
    }
}

impl<'a, 'tcx> dot::GraphWalk<'a, Node, Edge> for ConstraintGraph<'a, 'tcx> {
    fn nodes(&self) -> dot::Nodes<Node> {
        let mut set = FnvHashSet();
        for node in self.node_ids.keys() {
            set.insert(*node);
        }
        debug!("constraint graph has {} nodes", set.len());
        set.into_iter().collect()
    }
    fn edges(&self) -> dot::Edges<Edge> {
        debug!("constraint graph has {} edges", self.map.len());
        let mut v: Vec<_> = self.map.keys().map(|e| Edge::Constraint(*e)).collect();
        self.tcx.region_maps.each_encl_scope(|sub, sup| v.push(Edge::EnclScope(*sub, *sup)));
        debug!("region graph has {} edges", v.len());
        Cow::Owned(v)
    }
    fn source(&self, edge: &Edge) -> Node {
        let (n1, _) = edge_to_nodes(edge);
        debug!("edge {:?} has source {:?}", edge, n1);
        n1
    }
    fn target(&self, edge: &Edge) -> Node {
        let (_, n2) = edge_to_nodes(edge);
        debug!("edge {:?} has target {:?}", edge, n2);
        n2
    }
}

pub type ConstraintMap<'tcx> = FnvHashMap<Constraint, SubregionOrigin<'tcx>>;

fn dump_region_constraints_to<'a, 'tcx: 'a>(tcx: &'a ty::ctxt<'tcx>,
                                            map: &ConstraintMap<'tcx>,
                                            path: &str)
                                            -> io::Result<()> {
    debug!("dump_region_constraints map (len: {}) path: {}",
           map.len(),
           path);
    let g = ConstraintGraph::new(tcx, format!("region_constraints"), map);
    debug!("dump_region_constraints calling render");
    let mut v = Vec::new();
    dot::render(&g, &mut v).unwrap();
    File::create(path).and_then(|mut f| f.write_all(&v))
}
