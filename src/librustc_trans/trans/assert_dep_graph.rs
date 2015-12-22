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
//!
//! #[rustc_then_this_would_need("trans")] //~ ERROR OK
//! fn baz() { foo(); }
//! ```

use graphviz as dot;
use rustc::dep_graph::{DepGraphQuery, DepNode};
use rustc::middle::def_id::DefId;
use rustc::middle::ty;
use rustc_data_structures::fnv::{FnvHashMap, FnvHashSet};
use rustc_data_structures::graph::{Direction, INCOMING, OUTGOING, NodeIndex};
use rustc_front::hir;
use rustc_front::intravisit::Visitor;
use std::borrow::IntoCow;
use std::env;
use std::fs::File;
use std::io::Write;
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
                                FnvHashSet<(Span, DefId, DepNode)>>;
type TargetHashMap = FnvHashMap<InternedString,
                                FnvHashSet<(Span, InternedString, ast::NodeId, DepNode)>>;

struct IfThisChanged<'a, 'tcx:'a> {
    tcx: &'a ty::ctxt<'tcx>,
    if_this_changed: SourceHashMap,
    then_this_would_need: TargetHashMap,
}

impl<'a, 'tcx> IfThisChanged<'a, 'tcx> {
    fn process_attrs(&mut self, node_id: ast::NodeId, def_id: DefId) {
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
                                    .insert((attr.span, def_id, DepNode::Hir(def_id)));
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
                macro_rules! match_depnode_name {
                    ($input:expr, $def_id:expr, match { $($variant:ident,)* } else $y:expr) => {
                        match $input {
                            $(Some(stringify!($variant)) => DepNode::$variant($def_id),)*
                            _ => $y
                        }
                    }
                }
                let dep_node = match_depnode_name! {
                    dep_node_str, def_id, match {
                        CollectItem,
                        BorrowCheck,
                        TransCrateItem,
                        TypeckItemType,
                        TypeckItemBody,
                        ImplOrTraitItems,
                        ItemSignature,
                        FieldTy,
                        TraitItemDefIds,
                        InherentImpls,
                        ImplItems,
                        TraitImpls,
                        ReprHints,
                    } else {
                        self.tcx.sess.span_fatal(
                            attr.span,
                            &format!("unrecognized DepNode variant {:?}", dep_node_str));
                    }
                };
                let id = id.unwrap_or(InternedString::new(ID));
                self.then_this_would_need
                    .entry(id)
                    .or_insert(FnvHashSet())
                    .insert((attr.span, dep_node_interned.clone().unwrap(), node_id, dep_node));
            }
        }
    }
}

impl<'a, 'tcx> Visitor<'tcx> for IfThisChanged<'a, 'tcx> {
    fn visit_item(&mut self, item: &'tcx hir::Item) {
        let def_id = self.tcx.map.local_def_id(item.id);
        self.process_attrs(item.id, def_id);
    }
}

fn check_paths(tcx: &ty::ctxt,
               if_this_changed: &SourceHashMap,
               then_this_would_need: &TargetHashMap)
{
    // Return early here so as not to construct the query, which is not cheap.
    if if_this_changed.is_empty() {
        return;
    }
    let query = tcx.dep_graph.query();
    for (id, sources) in if_this_changed {
        let targets = match then_this_would_need.get(id) {
            Some(targets) => targets,
            None => {
                for &(source_span, _, _) in sources.iter().take(1) {
                    tcx.sess.span_err(
                        source_span,
                        &format!("no targets for id `{}`", id));
                }
                continue;
            }
        };

        for &(_, source_def_id, source_dep_node) in sources {
            let dependents = query.dependents(source_dep_node);
            for &(target_span, ref target_pass, _, ref target_dep_node) in targets {
                if !dependents.contains(&target_dep_node) {
                    tcx.sess.span_err(
                        target_span,
                        &format!("no path from `{}` to `{}`",
                                 tcx.item_path_str(source_def_id),
                                 target_pass));
                } else {
                    tcx.sess.span_err(
                        target_span,
                        &format!("OK"));
                }
            }
        }
    }
}

fn dump_graph(tcx: &ty::ctxt) {
    let path: String = env::var("RUST_DEP_GRAPH").unwrap_or_else(|_| format!("dep_graph"));
    let query = tcx.dep_graph.query();

    let nodes = match env::var("RUST_DEP_GRAPH_FILTER") {
        Ok(string) => {
            // Expect one of: "-> target", "source -> target", or "source ->".
            let parts: Vec<_> = string.split("->").collect();
            if parts.len() > 2 {
                panic!("Invalid RUST_DEP_GRAPH_FILTER: expected '[source] -> [target]'");
            }
            let sources = node_set(&query, &parts[0]);
            let targets = node_set(&query, &parts[1]);
            filter_nodes(&query, &sources, &targets)
        }
        Err(_) => {
            query.nodes()
                 .into_iter()
                 .collect()
        }
    };
    let edges = filter_edges(&query, &nodes);

    { // dump a .txt file with just the edges:
        let txt_path = format!("{}.txt", path);
        let mut file = File::create(&txt_path).unwrap();
        for &(source, target) in &edges {
            write!(file, "{:?} -> {:?}\n", source, target).unwrap();
        }
    }

    { // dump a .dot file in graphviz format:
        let dot_path = format!("{}.dot", path);
        let mut v = Vec::new();
        dot::render(&GraphvizDepGraph(nodes, edges), &mut v).unwrap();
        File::create(&dot_path).and_then(|mut f| f.write_all(&v)).unwrap();
    }
}

pub struct GraphvizDepGraph(FnvHashSet<DepNode>, Vec<(DepNode, DepNode)>);

impl<'a, 'tcx> dot::GraphWalk<'a, DepNode, (DepNode, DepNode)> for GraphvizDepGraph {
    fn nodes(&self) -> dot::Nodes<DepNode> {
        let nodes: Vec<_> = self.0.iter().cloned().collect();
        nodes.into_cow()
    }
    fn edges(&self) -> dot::Edges<(DepNode, DepNode)> {
        self.1[..].into_cow()
    }
    fn source(&self, edge: &(DepNode, DepNode)) -> DepNode {
        edge.0
    }
    fn target(&self, edge: &(DepNode, DepNode)) -> DepNode {
        edge.1
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

// Given an optional filter like `"x,y,z"`, returns either `None` (no
// filter) or the set of nodes whose labels contain all of those
// substrings.
fn node_set(query: &DepGraphQuery, filter: &str) -> Option<FnvHashSet<DepNode>> {
    debug!("node_set(filter={:?})", filter);

    if filter.trim().is_empty() {
        return None;
    }

    let filters: Vec<&str> = filter.split("&").map(|s| s.trim()).collect();

    debug!("node_set: filters={:?}", filters);

    Some(query.nodes()
         .into_iter()
         .filter(|n| {
             let s = format!("{:?}", n);
             filters.iter().all(|f| s.contains(f))
         })
        .collect())
}

fn filter_nodes(query: &DepGraphQuery,
                sources: &Option<FnvHashSet<DepNode>>,
                targets: &Option<FnvHashSet<DepNode>>)
                -> FnvHashSet<DepNode>
{
    if let &Some(ref sources) = sources {
        if let &Some(ref targets) = targets {
            walk_between(query, sources, targets)
        } else {
            walk_nodes(query, sources, OUTGOING)
        }
    } else if let &Some(ref targets) = targets {
        walk_nodes(query, targets, INCOMING)
    } else {
        query.nodes().into_iter().collect()
    }
}

fn walk_nodes(query: &DepGraphQuery,
              starts: &FnvHashSet<DepNode>,
              direction: Direction)
              -> FnvHashSet<DepNode>
{
    let mut set = FnvHashSet();
    for start in starts {
        debug!("walk_nodes: start={:?} outgoing?={:?}", start, direction == OUTGOING);
        if set.insert(*start) {
            let mut stack = vec![query.indices[start]];
            while let Some(index) = stack.pop() {
                for (_, edge) in query.graph.adjacent_edges(index, direction) {
                    let neighbor_index = edge.source_or_target(direction);
                    let neighbor = query.graph.node_data(neighbor_index);
                    if set.insert(*neighbor) {
                        stack.push(neighbor_index);
                    }
                }
            }
        }
    }
    set
}

fn walk_between(query: &DepGraphQuery,
                sources: &FnvHashSet<DepNode>,
                targets: &FnvHashSet<DepNode>)
                -> FnvHashSet<DepNode>
{
    // This is a bit tricky. We want to include a node only if it is:
    // (a) reachable from a source and (b) will reach a target. And we
    // have to be careful about cycles etc.  Luckily efficiency is not
    // a big concern!

    #[derive(Copy, Clone, PartialEq)]
    enum State { Undecided, Deciding, Included, Excluded }

    let mut node_states = vec![State::Undecided; query.graph.len_nodes()];

    for &target in targets {
        node_states[query.indices[&target].0] = State::Included;
    }

    for source in sources.iter().map(|n| query.indices[n]) {
        recurse(query, &mut node_states, source);
    }

    return query.nodes()
                .into_iter()
                .filter(|n| {
                    let index = query.indices[n];
                    node_states[index.0] == State::Included
                })
                .collect();

    fn recurse(query: &DepGraphQuery,
               node_states: &mut [State],
               node: NodeIndex)
               -> bool
    {
        match node_states[node.0] {
            // known to reach a target
            State::Included => return true,

            // known not to reach a target
            State::Excluded => return false,

            // backedge, not yet known, say false
            State::Deciding => return false,

            State::Undecided => { }
        }

        node_states[node.0] = State::Deciding;

        for neighbor_index in query.graph.successor_nodes(node) {
            if recurse(query, node_states, neighbor_index) {
                node_states[node.0] = State::Included;
            }
        }

        // if we didn't find a path to target, then set to excluded
        if node_states[node.0] == State::Deciding {
            node_states[node.0] = State::Excluded;
            false
        } else {
            assert!(node_states[node.0] == State::Included);
            true
        }
    }
}

fn filter_edges(query: &DepGraphQuery,
                nodes: &FnvHashSet<DepNode>)
                -> Vec<(DepNode, DepNode)>
{
    query.edges()
         .into_iter()
         .filter(|&(source, target)| nodes.contains(&source) && nodes.contains(&target))
         .collect()
}
