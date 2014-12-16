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
use super::Constraint;
use middle::infer::SubregionOrigin;
use middle::infer::region_inference::RegionVarBindings;
use session::config;
use util::nodemap::{FnvHashMap, FnvHashSet};
use util::ppaux::Repr;

use std::collections::hash_map::Vacant;
use std::io::{mod, File};
use std::os;
use std::sync::atomic;
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

    if !region_vars.tcx.sess.debugging_opt(config::PRINT_REGION_GRAPH) {
        return;
    }

    let requested_node : Option<ast::NodeId> =
        os::getenv("RUST_REGION_GRAPH_NODE").and_then(|s|from_str(s.as_slice()));

    if requested_node.is_some() && requested_node != Some(subject_node) {
        return;
    }

    let requested_output = os::getenv("RUST_REGION_GRAPH");
    debug!("requested_output: {} requested_node: {}",
           requested_output, requested_node);

    let output_path = {
        let output_template = match requested_output {
            Some(ref s) if s.as_slice() == "help" => {
                static PRINTED_YET : atomic::AtomicBool = atomic::INIT_ATOMIC_BOOL;
                if !PRINTED_YET.load(atomic::SeqCst) {
                    print_help_message();
                    PRINTED_YET.store(true, atomic::SeqCst);
                }
                return;
            }

            Some(other_path) => other_path,
            None => "/tmp/constraints.node%.dot".to_string(),
        };

        if output_template.len() == 0 {
            tcx.sess.bug("empty string provided as RUST_REGION_GRAPH");
        }

        if output_template.contains_char('%') {
            let mut new_str = String::new();
            for c in output_template.chars() {
                if c == '%' {
                    new_str.push_str(subject_node.to_string().as_slice());
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
    match dump_region_constraints_to(tcx, constraints, output_path.as_slice()) {
        Ok(()) => {}
        Err(e) => {
            let msg = format!("io error dumping region constraints: {}", e);
            region_vars.tcx.sess.err(msg.as_slice())
        }
    }
}

struct ConstraintGraph<'a, 'tcx: 'a> {
    tcx: &'a ty::ctxt<'tcx>,
    graph_name: String,
    map: &'a FnvHashMap<Constraint, SubregionOrigin<'tcx>>,
    node_ids: FnvHashMap<Node, uint>,
}

#[deriving(Clone, Hash, PartialEq, Eq, Show)]
enum Node {
    RegionVid(ty::RegionVid),
    Region(ty::Region),
}

type Edge = Constraint;

impl<'a, 'tcx> ConstraintGraph<'a, 'tcx> {
    fn new(tcx: &'a ty::ctxt<'tcx>,
           name: String,
           map: &'a ConstraintMap<'tcx>) -> ConstraintGraph<'a, 'tcx> {
        let mut i = 0;
        let mut node_ids = FnvHashMap::new();
        {
            let add_node = |node| {
                if let Vacant(e) = node_ids.entry(node) {
                    e.set(i);
                    i += 1;
                }
            };

            for (n1, n2) in map.keys().map(|c|constraint_to_nodes(c)) {
                add_node(n1);
                add_node(n2);
            }
        }

        ConstraintGraph { tcx: tcx,
                          graph_name: name,
                          map: map,
                          node_ids: node_ids }
    }
}

impl<'a, 'tcx> dot::Labeller<'a, Node, Edge> for ConstraintGraph<'a, 'tcx> {
    fn graph_id(&self) -> dot::Id {
        dot::Id::new(self.graph_name.as_slice()).unwrap()
    }
    fn node_id(&self, n: &Node) -> dot::Id {
        dot::Id::new(format!("node_{}", self.node_ids.get(n).unwrap())).unwrap()
    }
    fn node_label(&self, n: &Node) -> dot::LabelText {
        match *n {
            Node::RegionVid(n_vid) =>
                dot::LabelText::label(format!("{}", n_vid)),
            Node::Region(n_rgn) =>
                dot::LabelText::label(format!("{}", n_rgn.repr(self.tcx))),
        }
    }
    fn edge_label(&self, e: &Edge) -> dot::LabelText {
        dot::LabelText::label(format!("{}", self.map.get(e).unwrap().repr(self.tcx)))
    }
}

fn constraint_to_nodes(c: &Constraint) -> (Node, Node) {
    match *c {
        Constraint::ConstrainVarSubVar(rv_1, rv_2) => (Node::RegionVid(rv_1),
                                                       Node::RegionVid(rv_2)),
        Constraint::ConstrainRegSubVar(r_1, rv_2) => (Node::Region(r_1),
                                                      Node::RegionVid(rv_2)),
        Constraint::ConstrainVarSubReg(rv_1, r_2) => (Node::RegionVid(rv_1),
                                                      Node::Region(r_2)),
    }
}

impl<'a, 'tcx> dot::GraphWalk<'a, Node, Edge> for ConstraintGraph<'a, 'tcx> {
    fn nodes(&self) -> dot::Nodes<Node> {
        let mut set = FnvHashSet::new();
        for constraint in self.map.keys() {
            let (n1, n2) = constraint_to_nodes(constraint);
            set.insert(n1);
            set.insert(n2);
        }
        debug!("constraint graph has {} nodes", set.len());
        set.into_iter().collect()
    }
    fn edges(&self) -> dot::Edges<Edge> {
        debug!("constraint graph has {} edges", self.map.len());
        self.map.keys().map(|e|*e).collect()
    }
    fn source(&self, edge: &Edge) -> Node {
        let (n1, _) = constraint_to_nodes(edge);
        debug!("edge {} has source {}", edge, n1);
        n1
    }
    fn target(&self, edge: &Edge) -> Node {
        let (_, n2) = constraint_to_nodes(edge);
        debug!("edge {} has target {}", edge, n2);
        n2
    }
}

pub type ConstraintMap<'tcx> = FnvHashMap<Constraint, SubregionOrigin<'tcx>>;

fn dump_region_constraints_to<'a, 'tcx:'a >(tcx: &'a ty::ctxt<'tcx>,
                                            map: &ConstraintMap<'tcx>,
                                            path: &str) -> io::IoResult<()> {
    debug!("dump_region_constraints map (len: {}) path: {}", map.len(), path);
    let g = ConstraintGraph::new(tcx, format!("region_constraints"), map);
    let mut f = File::create(&Path::new(path));
    debug!("dump_region_constraints calling render");
    dot::render(&g, &mut f)
}
