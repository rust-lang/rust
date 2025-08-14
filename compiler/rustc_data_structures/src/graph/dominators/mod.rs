//! Finding the dominators in a control-flow graph.
//!
//! Algorithm based on Loukas Georgiadis,
//! "Linear-Time Algorithms for Dominators and Related Problems",
//! <https://www.cs.princeton.edu/techreports/2005/737.pdf>
//!
//! Additionally useful is the original Lengauer-Tarjan paper on this subject,
//! "A Fast Algorithm for Finding Dominators in a Flowgraph"
//! Thomas Lengauer and Robert Endre Tarjan.
//! <https://www.cs.princeton.edu/courses/archive/spr03/cs423/download/dominators.pdf>

use rustc_index::{Idx, IndexSlice, IndexVec};

use super::ControlFlowGraph;

#[cfg(test)]
mod tests;

struct PreOrderFrame<Iter> {
    pre_order_idx: PreorderIndex,
    iter: Iter,
}

rustc_index::newtype_index! {
    #[orderable]
    struct PreorderIndex {}
}

#[derive(Clone, Debug)]
pub struct Dominators<Node: Idx> {
    kind: Kind<Node>,
}

#[derive(Clone, Debug)]
enum Kind<Node: Idx> {
    /// A representation optimized for a small path graphs.
    Path,
    General(Inner<Node>),
}

pub fn dominators<G: ControlFlowGraph>(g: &G) -> Dominators<G::Node> {
    // We often encounter MIR bodies with 1 or 2 basic blocks. Special case the dominators
    // computation and representation for those cases.
    if is_small_path_graph(g) {
        Dominators { kind: Kind::Path }
    } else {
        Dominators { kind: Kind::General(dominators_impl(g)) }
    }
}

fn is_small_path_graph<G: ControlFlowGraph>(g: &G) -> bool {
    if g.start_node().index() != 0 {
        return false;
    }
    if g.num_nodes() == 1 {
        return true;
    }
    if g.num_nodes() == 2 {
        return g.successors(g.start_node()).any(|n| n.index() == 1);
    }
    false
}

fn dominators_impl<G: ControlFlowGraph>(graph: &G) -> Inner<G::Node> {
    // We allocate capacity for the full set of nodes, because most of the time
    // most of the nodes *are* reachable.
    let mut parent: IndexVec<PreorderIndex, PreorderIndex> =
        IndexVec::with_capacity(graph.num_nodes());

    let mut stack = vec![PreOrderFrame {
        pre_order_idx: PreorderIndex::ZERO,
        iter: graph.successors(graph.start_node()),
    }];
    let mut pre_order_to_real: IndexVec<PreorderIndex, G::Node> =
        IndexVec::with_capacity(graph.num_nodes());
    let mut real_to_pre_order: IndexVec<G::Node, Option<PreorderIndex>> =
        IndexVec::from_elem_n(None, graph.num_nodes());
    pre_order_to_real.push(graph.start_node());
    parent.push(PreorderIndex::ZERO); // the parent of the root node is the root for now.
    real_to_pre_order[graph.start_node()] = Some(PreorderIndex::ZERO);

    // Traverse the graph, collecting a number of things:
    //
    // * Preorder mapping (to it, and back to the actual ordering)
    // * Parents for each vertex in the preorder tree
    //
    // These are all done here rather than through one of the 'standard'
    // graph traversals to help make this fast.
    'recurse: while let Some(frame) = stack.last_mut() {
        for successor in frame.iter.by_ref() {
            if real_to_pre_order[successor].is_none() {
                let pre_order_idx = pre_order_to_real.push(successor);
                real_to_pre_order[successor] = Some(pre_order_idx);
                parent.push(frame.pre_order_idx);
                stack.push(PreOrderFrame { pre_order_idx, iter: graph.successors(successor) });

                continue 'recurse;
            }
        }

        stack.pop();
    }

    let reachable_vertices = pre_order_to_real.len();

    let mut idom = IndexVec::from_elem_n(PreorderIndex::ZERO, reachable_vertices);
    let mut semi = IndexVec::from_fn_n(std::convert::identity, reachable_vertices);
    let mut label = semi.clone();
    let mut bucket = IndexVec::from_elem_n(vec![], reachable_vertices);
    let mut lastlinked = None;

    // We loop over vertices in reverse preorder. This implements the pseudocode
    // of the simple Lengauer-Tarjan algorithm. A few key facts are noted here
    // which are helpful for understanding the code (full proofs and such are
    // found in various papers, including one cited at the top of this file).
    //
    // For each vertex w (which is not the root),
    //  * semi[w] is a proper ancestor of the vertex w (i.e., semi[w] != w)
    //  * idom[w] is an ancestor of semi[w] (i.e., idom[w] may equal semi[w])
    //
    // An immediate dominator of w (idom[w]) is a vertex v where v dominates w
    // and every other dominator of w dominates v. (Every vertex except the root has
    // a unique immediate dominator.)
    //
    // A semidominator for a given vertex w (semi[w]) is the vertex v with minimum
    // preorder number such that there exists a path from v to w in which all elements (other than w) have
    // preorder numbers greater than w (i.e., this path is not the tree path to
    // w).
    for w in (PreorderIndex::new(1)..PreorderIndex::new(reachable_vertices)).rev() {
        // Optimization: process buckets just once, at the start of the
        // iteration. Do not explicitly empty the bucket (even though it will
        // not be used again), to save some instructions.
        //
        // The bucket here contains the vertices whose semidominator is the
        // vertex w, which we are guaranteed to have found: all vertices who can
        // be semidominated by w must have a preorder number exceeding w, so
        // they have been placed in the bucket.
        //
        // We compute a partial set of immediate dominators here.
        for &v in bucket[w].iter() {
            // This uses the result of Lemma 5 from section 2 from the original
            // 1979 paper, to compute either the immediate or relative dominator
            // for a given vertex v.
            //
            // eval returns a vertex y, for which semi[y] is minimum among
            // vertices semi[v] +> y *> v. Note that semi[v] = w as we're in the
            // w bucket.
            //
            // Given such a vertex y, semi[y] <= semi[v] and idom[y] = idom[v].
            // If semi[y] = semi[v], though, idom[v] = semi[v].
            //
            // Using this, we can either set idom[v] to be:
            //  * semi[v] (i.e. w), if semi[y] is w
            //  * idom[y], otherwise
            //
            // We don't directly set to idom[y] though as it's not necessarily
            // known yet. The second preorder traversal will cleanup by updating
            // the idom for any that were missed in this pass.
            let y = eval(&mut parent, lastlinked, &semi, &mut label, v);
            idom[v] = if semi[y] < w { y } else { w };
        }

        // This loop computes the semi[w] for w.
        semi[w] = w;
        for v in graph.predecessors(pre_order_to_real[w]) {
            // TL;DR: Reachable vertices may have unreachable predecessors, so ignore any of them.
            //
            // Ignore blocks which are not connected to the entry block.
            //
            // The algorithm that was used to traverse the graph and build the
            // `pre_order_to_real` and `real_to_pre_order` vectors does so by
            // starting from the entry block and following the successors.
            // Therefore, any blocks not reachable from the entry block will be
            // set to `None` in the `pre_order_to_real` vector.
            //
            // For example, in this graph, A and B should be skipped:
            //
            //           ┌─────┐
            //           │     │
            //           └──┬──┘
            //              │
            //           ┌──▼──┐              ┌─────┐
            //           │     │              │  A  │
            //           └──┬──┘              └──┬──┘
            //              │                    │
            //      ┌───────┴───────┐            │
            //      │               │            │
            //   ┌──▼──┐         ┌──▼──┐      ┌──▼──┐
            //   │     │         │     │      │  B  │
            //   └──┬──┘         └──┬──┘      └──┬──┘
            //      │               └──────┬─────┘
            //   ┌──▼──┐                   │
            //   │     │                   │
            //   └──┬──┘                ┌──▼──┐
            //      │                   │     │
            //      │                   └─────┘
            //   ┌──▼──┐
            //   │     │
            //   └──┬──┘
            //      │
            //   ┌──▼──┐
            //   │     │
            //   └─────┘
            //
            // ...this may be the case if a MirPass modifies the CFG to remove
            // or rearrange certain blocks/edges.
            let Some(v) = real_to_pre_order[v] else { continue };

            // eval returns a vertex x from which semi[x] is minimum among
            // vertices semi[v] +> x *> v.
            //
            // From Lemma 4 from section 2, we know that the semidominator of a
            // vertex w is the minimum (by preorder number) vertex of the
            // following:
            //
            //  * direct predecessors of w with preorder number less than w
            //  * semidominators of u such that u > w and there exists (v, w)
            //    such that u *> v
            //
            // This loop therefore identifies such a minima. Note that any
            // semidominator path to w must have all but the first vertex go
            // through vertices numbered greater than w, so the reverse preorder
            // traversal we are using guarantees that all of the information we
            // might need is available at this point.
            //
            // The eval call will give us semi[x], which is either:
            //
            //  * v itself, if v has not yet been processed
            //  * A possible 'best' semidominator for w.
            let x = eval(&mut parent, lastlinked, &semi, &mut label, v);
            semi[w] = std::cmp::min(semi[w], semi[x]);
        }
        // semi[w] is now semidominator(w) and won't change any more.

        // Optimization: Do not insert into buckets if parent[w] = semi[w], as
        // we then immediately know the idom.
        //
        // If we don't yet know the idom directly, then push this vertex into
        // our semidominator's bucket, where it will get processed at a later
        // stage to compute its immediate dominator.
        let z = parent[w];
        if z != semi[w] {
            bucket[semi[w]].push(w);
        } else {
            idom[w] = z;
        }

        // Optimization: We share the parent array between processed and not
        // processed elements; lastlinked represents the divider.
        lastlinked = Some(w);
    }

    // Finalize the idoms for any that were not fully settable during initial
    // traversal.
    //
    // If idom[w] != semi[w] then we know that we've stored vertex y from above
    // into idom[w]. It is known to be our 'relative dominator', which means
    // that it's one of w's ancestors and has the same immediate dominator as w,
    // so use that idom.
    for w in PreorderIndex::new(1)..PreorderIndex::new(reachable_vertices) {
        if idom[w] != semi[w] {
            idom[w] = idom[idom[w]];
        }
    }

    let mut immediate_dominators = IndexVec::from_elem_n(None, graph.num_nodes());
    for (idx, node) in pre_order_to_real.iter_enumerated() {
        immediate_dominators[*node] = Some(pre_order_to_real[idom[idx]]);
    }

    let start_node = graph.start_node();
    immediate_dominators[start_node] = None;

    let time = compute_access_time(start_node, &immediate_dominators);

    Inner { immediate_dominators, time }
}

/// Evaluate the link-eval virtual forest, providing the currently minimum semi
/// value for the passed `node` (which may be itself).
///
/// This maintains that for every vertex v, `label[v]` is such that:
///
/// ```text
/// semi[eval(v)] = min { semi[label[u]] | root_in_forest(v) +> u *> v }
/// ```
///
/// where `+>` is a proper ancestor and `*>` is just an ancestor.
#[inline]
fn eval(
    ancestor: &mut IndexSlice<PreorderIndex, PreorderIndex>,
    lastlinked: Option<PreorderIndex>,
    semi: &IndexSlice<PreorderIndex, PreorderIndex>,
    label: &mut IndexSlice<PreorderIndex, PreorderIndex>,
    node: PreorderIndex,
) -> PreorderIndex {
    if is_processed(node, lastlinked) {
        compress(ancestor, lastlinked, semi, label, node);
        label[node]
    } else {
        node
    }
}

#[inline]
fn is_processed(v: PreorderIndex, lastlinked: Option<PreorderIndex>) -> bool {
    if let Some(ll) = lastlinked { v >= ll } else { false }
}

#[inline]
fn compress(
    ancestor: &mut IndexSlice<PreorderIndex, PreorderIndex>,
    lastlinked: Option<PreorderIndex>,
    semi: &IndexSlice<PreorderIndex, PreorderIndex>,
    label: &mut IndexSlice<PreorderIndex, PreorderIndex>,
    v: PreorderIndex,
) {
    assert!(is_processed(v, lastlinked));
    // Compute the processed list of ancestors
    //
    // We use a heap stack here to avoid recursing too deeply, exhausting the
    // stack space.
    let mut stack: smallvec::SmallVec<[_; 8]> = smallvec::smallvec![v];
    let mut u = ancestor[v];
    while is_processed(u, lastlinked) {
        stack.push(u);
        u = ancestor[u];
    }

    // Then in reverse order, popping the stack
    for &[v, u] in stack.array_windows().rev() {
        if semi[label[u]] < semi[label[v]] {
            label[v] = label[u];
        }
        ancestor[v] = ancestor[u];
    }
}

/// Tracks the list of dominators for each node.
#[derive(Clone, Debug)]
struct Inner<N: Idx> {
    // Even though we track only the immediate dominator of each node, it's
    // possible to get its full list of dominators by looking up the dominator
    // of each dominator.
    immediate_dominators: IndexVec<N, Option<N>>,
    time: IndexVec<N, Time>,
}

impl<Node: Idx> Dominators<Node> {
    /// Returns true if node is reachable from the start node.
    pub fn is_reachable(&self, node: Node) -> bool {
        match &self.kind {
            Kind::Path => true,
            Kind::General(g) => g.time[node].start != 0,
        }
    }

    /// Returns the immediate dominator of node, if any.
    pub fn immediate_dominator(&self, node: Node) -> Option<Node> {
        match &self.kind {
            Kind::Path => {
                if 0 < node.index() {
                    Some(Node::new(node.index() - 1))
                } else {
                    None
                }
            }
            Kind::General(g) => g.immediate_dominators[node],
        }
    }

    /// Returns true if `a` dominates `b`.
    ///
    /// # Panics
    ///
    /// Panics if `b` is unreachable.
    #[inline]
    pub fn dominates(&self, a: Node, b: Node) -> bool {
        match &self.kind {
            Kind::Path => a.index() <= b.index(),
            Kind::General(g) => {
                let a = g.time[a];
                let b = g.time[b];
                assert!(b.start != 0, "node {b:?} is not reachable");
                a.start <= b.start && b.finish <= a.finish
            }
        }
    }
}

/// Describes the number of vertices discovered at the time when processing of a particular vertex
/// started and when it finished. Both values are zero for unreachable vertices.
#[derive(Copy, Clone, Default, Debug)]
struct Time {
    start: u32,
    finish: u32,
}

fn compute_access_time<N: Idx>(
    start_node: N,
    immediate_dominators: &IndexSlice<N, Option<N>>,
) -> IndexVec<N, Time> {
    // Transpose the dominator tree edges, so that child nodes of vertex v are stored in
    // node[edges[v].start..edges[v].end].
    let mut edges: IndexVec<N, std::ops::Range<u32>> =
        IndexVec::from_elem(0..0, immediate_dominators);
    for &idom in immediate_dominators.iter() {
        if let Some(idom) = idom {
            edges[idom].end += 1;
        }
    }
    let mut m = 0;
    for e in edges.iter_mut() {
        m += e.end;
        e.start = m;
        e.end = m;
    }
    let mut node = IndexVec::from_elem_n(Idx::new(0), m.try_into().unwrap());
    for (i, &idom) in immediate_dominators.iter_enumerated() {
        if let Some(idom) = idom {
            edges[idom].start -= 1;
            node[edges[idom].start] = i;
        }
    }

    // Perform a depth-first search of the dominator tree. Record the number of vertices discovered
    // when vertex v is discovered first as time[v].start, and when its processing is finished as
    // time[v].finish.
    let mut time: IndexVec<N, Time> = IndexVec::from_elem(Time::default(), immediate_dominators);
    let mut stack = Vec::new();

    let mut discovered = 1;
    stack.push(start_node);
    time[start_node].start = discovered;

    while let Some(&i) = stack.last() {
        let e = &mut edges[i];
        if e.start == e.end {
            // Finish processing vertex i.
            time[i].finish = discovered;
            stack.pop();
        } else {
            let j = node[e.start];
            e.start += 1;
            // Start processing vertex j.
            discovered += 1;
            time[j].start = discovered;
            stack.push(j);
        }
    }

    time
}
