// Copyright 2012-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use rustc::dep_graph::{DepGraphQuery, DepNode};
use rustc::hir::def_id::DefId;
use rustc_data_structures::fx::FxHashMap;
use rustc_data_structures::bitvec::BitVector;
use rustc_data_structures::graph::{NodeIndex, Graph};

use super::hash::*;
use ich::Fingerprint;

/// A data-structure that makes it easy to enumerate the hashable
/// predecessors of any given dep-node.
pub struct Predecessors<'query> {
    // - Keys: dep-nodes that may have work-products, output meta-data
    //   nodes.
    // - Values: transitive predecessors of the key that are hashable
    //   (e.g., HIR nodes, input meta-data nodes)
    pub inputs: FxHashMap<&'query DepNode<DefId>, Vec<&'query DepNode<DefId>>>,

    // - Keys: some hashable node
    // - Values: the hash thereof
    pub hashes: FxHashMap<&'query DepNode<DefId>, Fingerprint>,
}

impl<'q> Predecessors<'q> {
    pub fn new(query: &'q DepGraphQuery<DefId>, hcx: &mut HashContext) -> Self {
        // Find nodes for which we want to know the full set of preds
        let tcx = hcx.tcx;
        let node_count = query.graph.len_nodes();

        // Set up some data structures the cache predecessor search needs:
        let mut visit_counts: Vec<u32> = Vec::new();
        let mut node_cache: Vec<Option<Box<[u32]>>> = Vec::new();
        visit_counts.resize(node_count, 0);
        node_cache.resize(node_count, None);
        let mut dfs_workspace1 = DfsWorkspace::new(node_count);
        let mut dfs_workspace2 = DfsWorkspace::new(node_count);

        let inputs: FxHashMap<_, _> = query
            .graph
            .all_nodes()
            .iter()
            .enumerate()
            .filter(|&(_, node)| match node.data {
                DepNode::WorkProduct(_) => true,
                DepNode::MetaData(ref def_id) => def_id.is_local(),

                // if -Z query-dep-graph is passed, save more extended data
                // to enable better unit testing
                DepNode::TypeckItemBody(_) |
                DepNode::TransCrateItem(_) => tcx.sess.opts.debugging_opts.query_dep_graph,

                _ => false,
            })
            .map(|(node_index, node)| {
                find_roots(&query.graph,
                           node_index as u32,
                           &mut visit_counts,
                           &mut node_cache[..],
                           HashContext::is_hashable,
                           &mut dfs_workspace1,
                           Some(&mut dfs_workspace2));

                let inputs: Vec<_> = dfs_workspace1.output.nodes.iter().map(|&i| {
                    query.graph.node_data(NodeIndex(i as usize))
                }).collect();

                (&node.data, inputs)
            })
            .collect();

        let mut hashes = FxHashMap();
        for input in inputs.values().flat_map(|v| v.iter().cloned()) {
            hashes.entry(input)
                  .or_insert_with(|| hcx.hash(input).unwrap());
        }

        Predecessors {
            inputs: inputs,
            hashes: hashes,
        }
    }
}

const CACHING_THRESHOLD: u32 = 60;

// Starting at `start_node`, this function finds this node's "roots", that is,
// anything that is hashable, in the dep-graph. It uses a simple depth-first
// search to achieve that. However, since some sub-graphs are traversed over
// and over again, the function also some caching built into it: Each time it
// visits a node it increases a counter for that node. If a node has been
// visited more often than CACHING_THRESHOLD, the function will allocate a
// cache entry in the `cache` array. This cache entry contains a flat list of
// all roots reachable from the given node. The next time the node is visited,
// the search can just add the contents of this array to the output instead of
// recursing further.
//
// The function takes two `DfsWorkspace` arguments. These contains some data
// structures that would be expensive to re-allocate all the time, so they are
// allocated once up-front. There are two of them because building a cache entry
// requires a recursive invocation of this function. Two are enough though,
// since function never recurses more than once.
fn find_roots<T, F>(graph: &Graph<T, ()>,
                    start_node: u32,
                    visit_counts: &mut [u32],
                    cache: &mut [Option<Box<[u32]>>],
                    is_root: F,
                    workspace: &mut DfsWorkspace,
                    mut sub_workspace: Option<&mut DfsWorkspace>)
    where F: Copy + Fn(&T) -> bool,
          T: ::std::fmt::Debug,
{
    workspace.visited.clear();
    workspace.output.clear();
    workspace.stack.clear();
    workspace.stack.push(start_node);

    loop {
        let node = match workspace.stack.pop() {
            Some(node) => node,
            None => return,
        };

        if !workspace.visited.insert(node as usize) {
            continue
        }

        if is_root(graph.node_data(NodeIndex(node as usize))) {
            // If this is a root, just add it to the output.
            workspace.output.insert(node);
        } else {
            if let Some(ref cached) = cache[node as usize] {
                for &n in &cached[..] {
                    workspace.output.insert(n);
                }
                // No need to recurse further from this node
                continue
            }

            visit_counts[node as usize] += 1;

            // If this node has been visited often enough to be cached ...
            if visit_counts[node as usize] > CACHING_THRESHOLD {
                // ... we are actually allowed to cache something, do so:
                if let Some(ref mut sub_workspace) = sub_workspace {
                    // Note that the following recursive invocation does never
                    // write to the cache (since we pass None as sub_workspace).
                    // This is intentional: The graph we are working with
                    // contains cycles and this prevent us from simply building
                    // our caches recursively on-demand.
                    // However, we can just do a regular, non-caching DFS to
                    // yield the set of roots and cache that.
                    find_roots(graph,
                               node,
                               visit_counts,
                               cache,
                               is_root,
                               sub_workspace,
                               None);

                    for &n in &sub_workspace.output.nodes {
                        workspace.output.insert(n);
                    }

                    cache[node as usize] = Some(sub_workspace.output
                                                             .nodes
                                                             .clone()
                                                             .into_boxed_slice());
                    // No need to recurse further from this node
                    continue
                }
            }

            for pred in graph.predecessor_nodes(NodeIndex(node as usize)) {
                workspace.stack.push(pred.node_id() as u32);
            }
        }
    }
}

struct DfsWorkspace {
    stack: Vec<u32>,
    visited: BitVector,
    output: NodeIndexSet,
}

impl DfsWorkspace {
    fn new(total_node_count: usize) -> DfsWorkspace {
        DfsWorkspace {
            stack: Vec::new(),
            visited: BitVector::new(total_node_count),
            output: NodeIndexSet::new(total_node_count),
        }
    }
}

struct NodeIndexSet {
    bitset: BitVector,
    nodes: Vec<u32>,
}

impl NodeIndexSet {
    fn new(total_node_count: usize) -> NodeIndexSet {
        NodeIndexSet {
            bitset: BitVector::new(total_node_count),
            nodes: Vec::new(),
        }
    }

    #[inline]
    fn clear(&mut self) {
        self.bitset.clear();
        self.nodes.clear();
    }

    #[inline]
    fn insert(&mut self, node: u32) {
        if self.bitset.insert(node as usize) {
            self.nodes.push(node)
        }
    }
}

#[test]
fn test_cached_dfs_acyclic() {

    //     0     1   2
    //     |      \ /
    //     3---+   |
    //     |   |   |
    //     |   |   |
    //     4   5   6
    //      \ / \ / \
    //       |   |   |
    //       7   8   9

    let mut g: Graph<bool, ()> = Graph::new();
    g.add_node(false);
    g.add_node(false);
    g.add_node(false);
    g.add_node(false);
    g.add_node(false);
    g.add_node(false);
    g.add_node(false);
    g.add_node(true);
    g.add_node(true);
    g.add_node(true);

    g.add_edge(NodeIndex(3), NodeIndex(0), ());
    g.add_edge(NodeIndex(4), NodeIndex(3), ());
    g.add_edge(NodeIndex(7), NodeIndex(4), ());
    g.add_edge(NodeIndex(5), NodeIndex(3), ());
    g.add_edge(NodeIndex(7), NodeIndex(5), ());
    g.add_edge(NodeIndex(8), NodeIndex(5), ());
    g.add_edge(NodeIndex(8), NodeIndex(6), ());
    g.add_edge(NodeIndex(9), NodeIndex(6), ());
    g.add_edge(NodeIndex(6), NodeIndex(1), ());
    g.add_edge(NodeIndex(6), NodeIndex(2), ());

    let mut ws1 = DfsWorkspace::new(g.len_nodes());
    let mut ws2 = DfsWorkspace::new(g.len_nodes());
    let mut visit_counts: Vec<_> = g.all_nodes().iter().map(|_| 0u32).collect();
    let mut cache: Vec<Option<Box<[u32]>>> = g.all_nodes().iter().map(|_| None).collect();

    fn is_root(x: &bool) -> bool { *x }

    for _ in 0 .. CACHING_THRESHOLD + 1 {
        find_roots(&g, 5, &mut visit_counts, &mut cache[..], is_root, &mut ws1, Some(&mut ws2));
        ws1.output.nodes.sort();
        assert_eq!(ws1.output.nodes, vec![7, 8]);

        find_roots(&g, 6, &mut visit_counts, &mut cache[..], is_root, &mut ws1, Some(&mut ws2));
        ws1.output.nodes.sort();
        assert_eq!(ws1.output.nodes, vec![8, 9]);

        find_roots(&g, 0, &mut visit_counts, &mut cache[..], is_root, &mut ws1, Some(&mut ws2));
        ws1.output.nodes.sort();
        assert_eq!(ws1.output.nodes, vec![7, 8]);

        find_roots(&g, 1, &mut visit_counts, &mut cache[..], is_root, &mut ws1, Some(&mut ws2));
        ws1.output.nodes.sort();
        assert_eq!(ws1.output.nodes, vec![8, 9]);

        find_roots(&g, 2, &mut visit_counts, &mut cache[..], is_root, &mut ws1, Some(&mut ws2));
        ws1.output.nodes.sort();
        assert_eq!(ws1.output.nodes, vec![8, 9]);

        find_roots(&g, 3, &mut visit_counts, &mut cache[..], is_root, &mut ws1, Some(&mut ws2));
        ws1.output.nodes.sort();
        assert_eq!(ws1.output.nodes, vec![7, 8]);

        find_roots(&g, 4, &mut visit_counts, &mut cache[..], is_root, &mut ws1, Some(&mut ws2));
        ws1.output.nodes.sort();
        assert_eq!(ws1.output.nodes, vec![7]);
    }
}

#[test]
fn test_cached_dfs_cyclic() {

    //    0       1 <---- 2       3
    //    ^       |       ^       ^
    //    |       v       |       |
    //    4 ----> 5 ----> 6 ----> 7
    //    ^       ^       ^       ^
    //    |       |       |       |
    //    8       9      10      11


    let mut g: Graph<bool, ()> = Graph::new();
    g.add_node(false);
    g.add_node(false);
    g.add_node(false);
    g.add_node(false);
    g.add_node(false);
    g.add_node(false);
    g.add_node(false);
    g.add_node(false);
    g.add_node(true);
    g.add_node(true);
    g.add_node(true);
    g.add_node(true);

    g.add_edge(NodeIndex( 4), NodeIndex(0), ());
    g.add_edge(NodeIndex( 8), NodeIndex(4), ());
    g.add_edge(NodeIndex( 4), NodeIndex(5), ());
    g.add_edge(NodeIndex( 1), NodeIndex(5), ());
    g.add_edge(NodeIndex( 9), NodeIndex(5), ());
    g.add_edge(NodeIndex( 5), NodeIndex(6), ());
    g.add_edge(NodeIndex( 6), NodeIndex(2), ());
    g.add_edge(NodeIndex( 2), NodeIndex(1), ());
    g.add_edge(NodeIndex(10), NodeIndex(6), ());
    g.add_edge(NodeIndex( 6), NodeIndex(7), ());
    g.add_edge(NodeIndex(11), NodeIndex(7), ());
    g.add_edge(NodeIndex( 7), NodeIndex(3), ());

    let mut ws1 = DfsWorkspace::new(g.len_nodes());
    let mut ws2 = DfsWorkspace::new(g.len_nodes());
    let mut visit_counts: Vec<_> = g.all_nodes().iter().map(|_| 0u32).collect();
    let mut cache: Vec<Option<Box<[u32]>>> = g.all_nodes().iter().map(|_| None).collect();

    fn is_root(x: &bool) -> bool { *x }

    for _ in 0 .. CACHING_THRESHOLD + 1 {
        find_roots(&g, 2, &mut visit_counts, &mut cache[..], is_root, &mut ws1, Some(&mut ws2));
        ws1.output.nodes.sort();
        assert_eq!(ws1.output.nodes, vec![8, 9, 10]);

        find_roots(&g, 3, &mut visit_counts, &mut cache[..], is_root, &mut ws1, Some(&mut ws2));
        ws1.output.nodes.sort();
        assert_eq!(ws1.output.nodes, vec![8, 9, 10, 11]);
    }
}
