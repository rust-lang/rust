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
//! exist in the graph. These checks run after trans, so they view the
//! the final state of the dependency graph. Note that there are
//! similar assertions found in `persist::dirty_clean` which check the
//! **initial** state of the dependency graph, just after it has been
//! loaded from disk.
//!
//! In this code, we report errors on each `rustc_if_this_changed`
//! annotation. If a path exists in all cases, then we would report
//! "all path(s) exist". Otherwise, we report: "no path to `foo`" for
//! each case where no path exists.  `compile-fail` tests can then be
//! used to check when paths exist or do not.
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
use rustc::dep_graph::debug::{DepNodeFilter, EdgeFilter};
use rustc::hir::def_id::DefId;
use rustc::ty::TyCtxt;
use rustc_data_structures::fnv::{FnvHashMap, FnvHashSet};
use rustc_data_structures::graph::{Direction, INCOMING, OUTGOING, NodeIndex};
use rustc::hir;
use rustc::hir::intravisit::Visitor;
use graphviz::IntoCow;
use std::env;
use std::fs::File;
use std::io::Write;
use syntax::ast;
use syntax::attr::AttrMetaMethods;
use syntax::parse::token::InternedString;
use syntax_pos::Span;

const IF_THIS_CHANGED: &'static str = "rustc_if_this_changed";
const THEN_THIS_WOULD_NEED: &'static str = "rustc_then_this_would_need";
const ID: &'static str = "id";

pub fn assert_dep_graph<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>) {
    let _ignore = tcx.dep_graph.in_ignore();

    if tcx.sess.opts.debugging_opts.dump_dep_graph {
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

    if !if_this_changed.is_empty() || !then_this_would_need.is_empty() {
        assert!(tcx.sess.opts.debugging_opts.query_dep_graph,
                "cannot use the `#[{}]` or `#[{}]` annotations \
                 without supplying `-Z query-dep-graph`",
                IF_THIS_CHANGED, THEN_THIS_WOULD_NEED);
    }

    // Check paths.
    check_paths(tcx, &if_this_changed, &then_this_would_need);
}

type SourceHashMap =
    FnvHashMap<InternedString,
               FnvHashSet<(Span, DefId, DepNode<DefId>)>>;
type TargetHashMap =
    FnvHashMap<InternedString,
               FnvHashSet<(Span, InternedString, ast::NodeId, DepNode<DefId>)>>;

struct IfThisChanged<'a, 'tcx:'a> {
    tcx: TyCtxt<'a, 'tcx, 'tcx>,
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
                        ast::MetaItemKind::Word(ref s) if id.is_none() => id = Some(s.clone()),
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
                        ast::MetaItemKind::Word(ref s) if dep_node_interned.is_none() =>
                            dep_node_interned = Some(s.clone()),
                        ast::MetaItemKind::Word(ref s) if id.is_none() =>
                            id = Some(s.clone()),
                        _ => {
                            self.tcx.sess.span_err(
                                meta_item.span,
                                &format!("unexpected meta-item {:?}", meta_item.node));
                        }
                    }
                }
                let dep_node = match dep_node_interned {
                    Some(ref n) => {
                        match DepNode::from_label_string(&n[..], def_id) {
                            Ok(n) => n,
                            Err(()) => {
                                self.tcx.sess.span_fatal(
                                    attr.span,
                                    &format!("unrecognized DepNode variant {:?}", n));
                            }
                        }
                    }
                    None => {
                        self.tcx.sess.span_fatal(
                            attr.span,
                            &format!("missing DepNode variant"));
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

fn check_paths<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>,
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

        for &(_, source_def_id, ref source_dep_node) in sources {
            let dependents = query.transitive_successors(source_dep_node);
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

fn dump_graph(tcx: TyCtxt) {
    let path: String = env::var("RUST_DEP_GRAPH").unwrap_or_else(|_| format!("dep_graph"));
    let query = tcx.dep_graph.query();

    let nodes = match env::var("RUST_DEP_GRAPH_FILTER") {
        Ok(string) => {
            // Expect one of: "-> target", "source -> target", or "source ->".
            let edge_filter = EdgeFilter::new(&string).unwrap_or_else(|e| {
                bug!("invalid filter: {}", e)
            });
            let sources = node_set(&query, &edge_filter.source);
            let targets = node_set(&query, &edge_filter.target);
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
        for &(ref source, ref target) in &edges {
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

pub struct GraphvizDepGraph<'q>(FnvHashSet<&'q DepNode<DefId>>,
                                Vec<(&'q DepNode<DefId>, &'q DepNode<DefId>)>);

impl<'a, 'tcx, 'q> dot::GraphWalk<'a> for GraphvizDepGraph<'q> {
    type Node = &'q DepNode<DefId>;
    type Edge = (&'q DepNode<DefId>, &'q DepNode<DefId>);
    fn nodes(&self) -> dot::Nodes<&'q DepNode<DefId>> {
        let nodes: Vec<_> = self.0.iter().cloned().collect();
        nodes.into_cow()
    }
    fn edges(&self) -> dot::Edges<(&'q DepNode<DefId>, &'q DepNode<DefId>)> {
        self.1[..].into_cow()
    }
    fn source(&self, edge: &(&'q DepNode<DefId>, &'q DepNode<DefId>)) -> &'q DepNode<DefId> {
        edge.0
    }
    fn target(&self, edge: &(&'q DepNode<DefId>, &'q DepNode<DefId>)) -> &'q DepNode<DefId> {
        edge.1
    }
}

impl<'a, 'tcx, 'q> dot::Labeller<'a> for GraphvizDepGraph<'q> {
    type Node = &'q DepNode<DefId>;
    type Edge = (&'q DepNode<DefId>, &'q DepNode<DefId>);
    fn graph_id(&self) -> dot::Id {
        dot::Id::new("DependencyGraph").unwrap()
    }
    fn node_id(&self, n: &&'q DepNode<DefId>) -> dot::Id {
        let s: String =
            format!("{:?}", n).chars()
                              .map(|c| if c == '_' || c.is_alphanumeric() { c } else { '_' })
                              .collect();
        debug!("n={:?} s={:?}", n, s);
        dot::Id::new(s).unwrap()
    }
    fn node_label(&self, n: &&'q DepNode<DefId>) -> dot::LabelText {
        dot::LabelText::label(format!("{:?}", n))
    }
}

// Given an optional filter like `"x,y,z"`, returns either `None` (no
// filter) or the set of nodes whose labels contain all of those
// substrings.
fn node_set<'q>(query: &'q DepGraphQuery<DefId>, filter: &DepNodeFilter)
                -> Option<FnvHashSet<&'q DepNode<DefId>>>
{
    debug!("node_set(filter={:?})", filter);

    if filter.accepts_all() {
        return None;
    }

    Some(query.nodes().into_iter().filter(|n| filter.test(n)).collect())
}

fn filter_nodes<'q>(query: &'q DepGraphQuery<DefId>,
                    sources: &Option<FnvHashSet<&'q DepNode<DefId>>>,
                    targets: &Option<FnvHashSet<&'q DepNode<DefId>>>)
                    -> FnvHashSet<&'q DepNode<DefId>>
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

fn walk_nodes<'q>(query: &'q DepGraphQuery<DefId>,
                  starts: &FnvHashSet<&'q DepNode<DefId>>,
                  direction: Direction)
                  -> FnvHashSet<&'q DepNode<DefId>>
{
    let mut set = FnvHashSet();
    for &start in starts {
        debug!("walk_nodes: start={:?} outgoing?={:?}", start, direction == OUTGOING);
        if set.insert(start) {
            let mut stack = vec![query.indices[start]];
            while let Some(index) = stack.pop() {
                for (_, edge) in query.graph.adjacent_edges(index, direction) {
                    let neighbor_index = edge.source_or_target(direction);
                    let neighbor = query.graph.node_data(neighbor_index);
                    if set.insert(neighbor) {
                        stack.push(neighbor_index);
                    }
                }
            }
        }
    }
    set
}

fn walk_between<'q>(query: &'q DepGraphQuery<DefId>,
                    sources: &FnvHashSet<&'q DepNode<DefId>>,
                    targets: &FnvHashSet<&'q DepNode<DefId>>)
                    -> FnvHashSet<&'q DepNode<DefId>>
{
    // This is a bit tricky. We want to include a node only if it is:
    // (a) reachable from a source and (b) will reach a target. And we
    // have to be careful about cycles etc.  Luckily efficiency is not
    // a big concern!

    #[derive(Copy, Clone, PartialEq)]
    enum State { Undecided, Deciding, Included, Excluded }

    let mut node_states = vec![State::Undecided; query.graph.len_nodes()];

    for &target in targets {
        node_states[query.indices[target].0] = State::Included;
    }

    for source in sources.iter().map(|&n| query.indices[n]) {
        recurse(query, &mut node_states, source);
    }

    return query.nodes()
                .into_iter()
                .filter(|&n| {
                    let index = query.indices[n];
                    node_states[index.0] == State::Included
                })
                .collect();

    fn recurse(query: &DepGraphQuery<DefId>,
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

fn filter_edges<'q>(query: &'q DepGraphQuery<DefId>,
                    nodes: &FnvHashSet<&'q DepNode<DefId>>)
                    -> Vec<(&'q DepNode<DefId>, &'q DepNode<DefId>)>
{
    query.edges()
         .into_iter()
         .filter(|&(source, target)| nodes.contains(source) && nodes.contains(target))
         .collect()
}
