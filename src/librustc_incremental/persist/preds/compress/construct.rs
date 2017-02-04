// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Second phase. Construct new graph. The previous phase has
//! converted the input graph into a DAG by detecting and unifying
//! cycles. It provides us with the following (which is a
//! representation of the DAG):
//!
//! - SCCs, in the form of a union-find repr that can convert each node to
//!   its *cycle head* (an arbitrarly chosen representative from the cycle)
//! - a vector of *leaf nodes*, just a convenience
//! - a vector of *parents* for each node (in some cases, nodes have no parents,
//!   or their parent is another member of same cycle; in that case, the vector
//!   will be stored `v[i] == i`, after canonicalization)
//! - a vector of *cross edges*, meaning add'l edges between graphs nodes beyond
//!   the parents.

use rustc_data_structures::fx::FxHashMap;

use super::*;

pub(super) fn construct_graph<'g, N, I, O>(r: &mut GraphReduce<'g, N, I, O>, dag: Dag)
                                           -> Reduction<'g, N>
    where N: Debug + Clone, I: Fn(&N) -> bool, O: Fn(&N) -> bool,
{
    let Dag { parents: old_parents, input_nodes, output_nodes, cross_edges } = dag;
    let in_graph = r.in_graph;

    debug!("construct_graph");

    // Create a canonical list of edges; this includes both parent and
    // cross-edges. We store this in `(target -> Vec<source>)` form.
    // We call the first edge to any given target its "parent".
    let mut edges = FxHashMap();
    let old_parent_edges = old_parents.iter().cloned().zip((0..).map(NodeIndex));
    for (source, target) in old_parent_edges.chain(cross_edges) {
        debug!("original edge `{:?} -rf-> {:?}`",
               in_graph.node_data(source),
               in_graph.node_data(target));
        let source = r.cycle_head(source);
        let target = r.cycle_head(target);
        if source != target {
            let v = edges.entry(target).or_insert(vec![]);
            if !v.contains(&source) {
                debug!("edge `{:?} -rf-> {:?}` is edge #{} with that target",
                       in_graph.node_data(source),
                       in_graph.node_data(target),
                       v.len());
                v.push(source);
            }
        }
    }
    let parent = |ni: NodeIndex| -> NodeIndex {
        edges[&ni][0]
    };

    // `retain_map`: a map of those nodes that we will want to
    // *retain* in the ultimate graph; the key is the node index in
    // the old graph, the value is the node index in the new
    // graph. These are nodes in the following categories:
    //
    // - inputs
    // - work-products
    // - targets of a cross-edge
    //
    // The first two categories hopefully make sense. We want the
    // inputs so we can compare hashes later. We want the
    // work-products so we can tell precisely when a given
    // work-product is invalidated. But the last one isn't strictly
    // needed; we keep cross-target edges so as to minimize the total
    // graph size.
    //
    // Consider a graph like:
    //
    //     WP0 -rf-> Y
    //     WP1 -rf-> Y
    //     Y -rf-> INPUT0
    //     Y -rf-> INPUT1
    //     Y -rf-> INPUT2
    //     Y -rf-> INPUT3
    //
    // Now if we were to remove Y, we would have a total of 8 edges: both WP0 and WP1
    // depend on INPUT0...INPUT3. As it is, we have 6 edges.
    //
    // NB: The current rules are not optimal. For example, given this
    // input graph:
    //
    //     OUT0 -rf-> X
    //     OUT1 -rf-> X
    //     X -rf -> INPUT0
    //
    // we will preserve X because it has two "consumers" (OUT0 and
    // OUT1).  We could as easily skip it, but we'd have to tally up
    // the number of input nodes that it (transitively) reaches, and I
    // was too lazy to do so. This is the unit test `suboptimal`.

    let mut retain_map = FxHashMap();
    let mut new_graph = Graph::new();

    {
        // Start by adding start-nodes and inputs.
        let retained_nodes = output_nodes.iter().chain(&input_nodes).map(|&n| r.cycle_head(n));

        // Next add in targets of cross-edges. Due to the canonicalization,
        // some of these may be self-edges or may may duplicate the parent
        // edges, so ignore those.
        let retained_nodes = retained_nodes.chain(
            edges.iter()
                 .filter(|&(_, ref sources)| sources.len() > 1)
                 .map(|(&target, _)| target));

        // Now create the new graph, adding in the entries from the map.
        for n in retained_nodes {
            retain_map.entry(n)
                      .or_insert_with(|| {
                          let data = in_graph.node_data(n);
                          debug!("retaining node `{:?}`", data);
                          new_graph.add_node(data)
                      });
        }
    }

    // Given a cycle-head `ni`, converts it to the closest parent that has
    // been retained in the output graph.
    let retained_parent = |mut ni: NodeIndex| -> NodeIndex {
        loop {
            debug!("retained_parent({:?})", in_graph.node_data(ni));
            match retain_map.get(&ni) {
                Some(&v) => return v,
                None => ni = parent(ni),
            }
        }
    };

    // Now add in the edges into the graph.
    for (&target, sources) in &edges {
        if let Some(&r_target) = retain_map.get(&target) {
            debug!("adding edges that target `{:?}`", in_graph.node_data(target));
            for &source in sources {
                debug!("new edge `{:?} -rf-> {:?}`",
                       in_graph.node_data(source),
                       in_graph.node_data(target));
                let r_source = retained_parent(source);

                // NB. In the input graph, we have `a -> b` if b
                // **reads from** a. But in the terminology of this
                // code, we would describe that edge as `b -> a`,
                // because we have edges *from* outputs *to* inputs.
                // Therefore, when we create our new graph, we have to
                // reverse the edge.
                new_graph.add_edge(r_target, r_source, ());
            }
        } else {
            assert_eq!(sources.len(), 1);
        }
    }

    // One complication. In some cases, output nodes *may* participate in
    // cycles. An example:
    //
    //             [HIR0]                    [HIR1]
    //               |                         |
    //               v                         v
    //      TypeckClosureBody(X) -> ItemSignature(X::SomeClosureInX)
    //            |  ^                         | |
    //            |  +-------------------------+ |
    //            |                              |
    //            v                              v
    //           Foo                            Bar
    //
    // In these cases, the output node may not wind up as the head
    // of the cycle, in which case it would be absent from the
    // final graph. We don't wish this to happen, therefore we go
    // over the list of output nodes again and check for any that
    // are not their own cycle-head. If we find such a node, we
    // add it to the graph now with an edge from the cycle head.
    // So the graph above could get transformed into this:
    //
    //                                    [HIR0, HIR1]
    //                                         |
    //                                         v
    //      TypeckClosureBody(X)    ItemSignature(X::SomeClosureInX)
    //               ^                         | |
    //               +-------------------------+ |
    //                                           v
    //                                       [Foo, Bar]
    //
    // (Note that all the edges here are "read-by" edges, not
    // "reads-from" edges.)
    for &output_node in &output_nodes {
        let head = r.cycle_head(output_node);
        if output_node == head {
            assert!(retain_map.contains_key(&output_node));
        } else {
            assert!(!retain_map.contains_key(&output_node));
            let output_data = in_graph.node_data(output_node);
            let new_node = new_graph.add_node(output_data);
            let new_head_node = retain_map[&head];
            new_graph.add_edge(new_head_node, new_node, ());
        }
    }

    // Finally, prepare a list of the input node indices as found in
    // the new graph. Note that since all input nodes are leaves in
    // the graph, they should never participate in a cycle.
    let input_nodes =
        input_nodes.iter()
                   .map(|&n| {
                       assert_eq!(r.cycle_head(n), n, "input node participating in a cycle");
                       retain_map[&n]
                   })
                   .collect();

    Reduction { graph: new_graph, input_nodes: input_nodes }
}

