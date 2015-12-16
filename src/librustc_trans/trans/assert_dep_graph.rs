// Copyright 2012-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! This pass is only used for the UNIT TESTS and DEBUGGING NEEDS
//! around dependency graph construction. It serves two purposes; it
//! will dump graphs in graphviz form to disk, and it searches for
//! `#[rustc_if_this_changed]` and `#[rustc_then_this_would_need]`
//! annotations. These annotations can be used to test whether paths
//! exist in the graph. We report errors on each
//! `rustc_if_this_changed` annotation. If a path exists in all
//! cases, then we would report "all path(s) exist". Otherwise, we
//! report: "no path to `foo`" for each case where no path exists.
//! `compile-fail` tests can then be used to check when paths exist or
//! do not.
//!
//! The full form of the `rustc_if_this_changed` annotation is
//! `#[rustc_if_this_changed(id)]`. The `"id"` is optional and
//! defaults to `"id"` if omitted.
//!
//! Example:
//!
//! ```
//! #[rustc_if_this_changed]
//! fn foo() { }
//!
//! #[rustc_then_this_would_need("trans")] //~ ERROR no path from `foo`
//! fn bar() { }
//! ```

use graphviz as dot;
use rustc::dep_graph::{DepGraph, DepNode};
use rustc::middle::ty;
use rustc_data_structures::fnv::{FnvHashMap, FnvHashSet};
use rustc_front::hir;
use rustc_front::intravisit::Visitor;
use std::borrow::IntoCow;
use std::env;
use std::fs::File;
use std::io::Write;
use std::rc::Rc;
use syntax::ast;
use syntax::attr::AttrMetaMethods;
use syntax::codemap::Span;
use syntax::parse::token::InternedString;

const IF_THIS_CHANGED: &'static str = "rustc_if_this_changed";
const THEN_THIS_WOULD_NEED: &'static str = "rustc_then_this_would_need";
const ID: &'static str = "id";

pub fn assert_dep_graph(tcx: &ty::ctxt) {
    let _ignore = tcx.dep_graph.in_ignore();

    if tcx.sess.opts.dump_dep_graph {
        dump_graph(tcx);
    }

    // Find annotations supplied by user (if any).
    let (if_this_changed, then_this_would_need) = {
        let mut visitor = IfThisChanged { tcx: tcx,
                                          if_this_changed: FnvHashMap(),
                                          then_this_would_need: FnvHashMap() };
        tcx.map.krate().visit_all_items(&mut visitor);
        (visitor.if_this_changed, visitor.then_this_would_need)
    };

    // Check paths.
    check_paths(tcx, &if_this_changed, &then_this_would_need);
}

type SourceHashMap = FnvHashMap<InternedString,
                                FnvHashSet<(Span, DepNode)>>;
type TargetHashMap = FnvHashMap<InternedString,
                                FnvHashSet<(Span, InternedString, ast::NodeId, DepNode)>>;

struct IfThisChanged<'a, 'tcx:'a> {
    tcx: &'a ty::ctxt<'tcx>,
    if_this_changed: SourceHashMap,
    then_this_would_need: TargetHashMap,
}

impl<'a, 'tcx> Visitor<'tcx> for IfThisChanged<'a, 'tcx> {
    fn visit_item(&mut self, item: &'tcx hir::Item) {
        let def_id = self.tcx.map.local_def_id(item.id);
        for attr in self.tcx.get_attrs(def_id).iter() {
            if attr.check_name(IF_THIS_CHANGED) {
                let mut id = None;
                for meta_item in attr.meta_item_list().unwrap_or_default() {
                    match meta_item.node {
                        ast::MetaWord(ref s) if id.is_none() => id = Some(s.clone()),
                        _ => {
                            self.tcx.sess.span_err(
                                meta_item.span,
                                &format!("unexpected meta-item {:?}", meta_item.node));
                        }
                    }
                }
                let id = id.unwrap_or(InternedString::new(ID));
                self.if_this_changed.entry(id)
                                    .or_insert(FnvHashSet())
                                    .insert((attr.span, DepNode::Hir(def_id)));
            } else if attr.check_name(THEN_THIS_WOULD_NEED) {
                let mut dep_node_interned = None;
                let mut id = None;
                for meta_item in attr.meta_item_list().unwrap_or_default() {
                    match meta_item.node {
                        ast::MetaWord(ref s) if dep_node_interned.is_none() =>
                            dep_node_interned = Some(s.clone()),
                        ast::MetaWord(ref s) if id.is_none() =>
                            id = Some(s.clone()),
                        _ => {
                            self.tcx.sess.span_err(
                                meta_item.span,
                                &format!("unexpected meta-item {:?}", meta_item.node));
                        }
                    }
                }
                let dep_node_str = dep_node_interned.as_ref().map(|s| &**s);
                let dep_node = match dep_node_str {
                    Some("BorrowCheck") => DepNode::BorrowCheck(def_id),
                    Some("TransCrateItem") => DepNode::TransCrateItem(def_id),
                    Some("TypeckItemType") => DepNode::TypeckItemType(def_id),
                    Some("TypeckItemBody") => DepNode::TypeckItemBody(def_id),
                    _ => {
                        self.tcx.sess.span_fatal(
                            attr.span,
                            &format!("unrecognized pass {:?}", dep_node_str));
                    }
                };
                let id = id.unwrap_or(InternedString::new(ID));
                self.then_this_would_need
                    .entry(id)
                    .or_insert(FnvHashSet())
                    .insert((attr.span, dep_node_interned.clone().unwrap(), item.id, dep_node));
            }
        }
    }
}

fn check_paths(tcx: &ty::ctxt,
               if_this_changed: &SourceHashMap,
               then_this_would_need: &TargetHashMap)
{
    for (id, sources) in if_this_changed {
        let targets = match then_this_would_need.get(id) {
            Some(targets) => targets,
            None => {
                for &(source_span, _) in sources.iter().take(1) {
                    tcx.sess.span_err(
                        source_span,
                        &format!("no targets for id `{}`", id));
                }
                continue;
            }
        };

        for &(source_span, ref source_dep_node) in sources {
            let dependents = tcx.dep_graph.dependents(source_dep_node.clone());
            for &(_, ref target_pass, target_node_id, ref target_dep_node) in targets {
                if !dependents.contains(&target_dep_node) {
                    let target_def_id = tcx.map.local_def_id(target_node_id);
                    tcx.sess.span_err(
                        source_span,
                        &format!("no path to {} for `{}`",
                                 target_pass,
                                 tcx.item_path_str(target_def_id)));
                }
            }
        }
    }
}

fn dump_graph(tcx: &ty::ctxt) {
    let path: String = env::var("RUST_DEP_GRAPH").unwrap_or_else(|_| format!("/tmp/dep_graph.dot"));
    let mut v = Vec::new();
    dot::render(&GraphvizDepGraph(tcx.dep_graph.clone()), &mut v).unwrap();
    File::create(&path).and_then(|mut f| f.write_all(&v)).unwrap();
}

pub struct GraphvizDepGraph(Rc<DepGraph>);

impl<'a, 'tcx> dot::GraphWalk<'a, DepNode, (DepNode, DepNode)> for GraphvizDepGraph {
    fn nodes(&self) -> dot::Nodes<DepNode> {
        self.0.nodes().into_cow()
    }
    fn edges(&self) -> dot::Edges<(DepNode, DepNode)> {
        self.0.edges().into_cow()
    }
    fn source(&self, edge: &(DepNode, DepNode)) -> DepNode {
        edge.0.clone()
    }
    fn target(&self, edge: &(DepNode, DepNode)) -> DepNode {
        edge.1.clone()
    }
}

impl<'a, 'tcx> dot::Labeller<'a, DepNode, (DepNode, DepNode)> for GraphvizDepGraph {
    fn graph_id(&self) -> dot::Id {
        dot::Id::new("DependencyGraph").unwrap()
    }
    fn node_id(&self, n: &DepNode) -> dot::Id {
        let s: String =
            format!("{:?}", n).chars()
                              .map(|c| if c == '_' || c.is_alphanumeric() { c } else { '_' })
                              .collect();
        debug!("n={:?} s={:?}", n, s);
        dot::Id::new(s).unwrap()
    }
    fn node_label(&self, n: &DepNode) -> dot::LabelText {
        dot::LabelText::label(format!("{:?}", n))
    }
}
