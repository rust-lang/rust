#![cfg(test)]

use crate::graph::test::TestGraph;
use super::*;

#[test]
fn diamond() {
    let graph = TestGraph::new(0, &[(0, 1), (0, 2), (1, 3), (2, 3)]);
    let sccs: Sccs<_, usize> = Sccs::new(&graph);
    assert_eq!(sccs.num_sccs(), 4);
    assert_eq!(sccs.num_sccs(), 4);
}

#[test]
fn test_big_scc() {
    // The order in which things will be visited is important to this
    // test.
    //
    // We will visit:
    //
    // 0 -> 1 -> 2 -> 0
    //
    // and at this point detect a cycle. 2 will return back to 1 which
    // will visit 3. 3 will visit 2 before the cycle is complete, and
    // hence it too will return a cycle.

    /*
+-> 0
|   |
|   v
|   1 -> 3
|   |    |
|   v    |
+-- 2 <--+
     */
    let graph = TestGraph::new(0, &[
        (0, 1),
        (1, 2),
        (1, 3),
        (2, 0),
        (3, 2),
    ]);
    let sccs: Sccs<_, usize> = Sccs::new(&graph);
    assert_eq!(sccs.num_sccs(), 1);
}

#[test]
fn test_three_sccs() {
    /*
    0
    |
    v
+-> 1    3
|   |    |
|   v    |
+-- 2 <--+
     */
    let graph = TestGraph::new(0, &[
        (0, 1),
        (1, 2),
        (2, 1),
        (3, 2),
    ]);
    let sccs: Sccs<_, usize> = Sccs::new(&graph);
    assert_eq!(sccs.num_sccs(), 3);
    assert_eq!(sccs.scc(0), 1);
    assert_eq!(sccs.scc(1), 0);
    assert_eq!(sccs.scc(2), 0);
    assert_eq!(sccs.scc(3), 2);
    assert_eq!(sccs.successors(0), &[]);
    assert_eq!(sccs.successors(1), &[0]);
    assert_eq!(sccs.successors(2), &[0]);
}

#[test]
fn test_find_state_2() {
    // The order in which things will be visited is important to this
    // test. It tests part of the `find_state` behavior. Here is the
    // graph:
    //
    //
    //       /----+
    //     0 <--+ |
    //     |    | |
    //     v    | |
    // +-> 1 -> 3 4
    // |   |      |
    // |   v      |
    // +-- 2 <----+

    let graph = TestGraph::new(0, &[
        (0, 1),
        (0, 4),
        (1, 2),
        (1, 3),
        (2, 1),
        (3, 0),
        (4, 2),
    ]);

    // For this graph, we will start in our DFS by visiting:
    //
    // 0 -> 1 -> 2 -> 1
    //
    // and at this point detect a cycle. The state of 2 will thus be
    // `InCycleWith { 1 }`.  We will then visit the 1 -> 3 edge, which
    // will attempt to visit 0 as well, thus going to the state
    // `InCycleWith { 0 }`. Finally, node 1 will complete; the lowest
    // depth of any successor was 3 which had depth 0, and thus it
    // will be in the state `InCycleWith { 3 }`.
    //
    // When we finally traverse the `0 -> 4` edge and then visit node 2,
    // the states of the nodes are:
    //
    // 0 BeingVisited { 0 }
    // 1 InCycleWith { 3 }
    // 2 InCycleWith { 1 }
    // 3 InCycleWith { 0 }
    //
    // and hence 4 will traverse the links, finding an ultimate depth of 0.
    // If will also collapse the states to the following:
    //
    // 0 BeingVisited { 0 }
    // 1 InCycleWith { 3 }
    // 2 InCycleWith { 1 }
    // 3 InCycleWith { 0 }

    let sccs: Sccs<_, usize> = Sccs::new(&graph);
    assert_eq!(sccs.num_sccs(), 1);
    assert_eq!(sccs.scc(0), 0);
    assert_eq!(sccs.scc(1), 0);
    assert_eq!(sccs.scc(2), 0);
    assert_eq!(sccs.scc(3), 0);
    assert_eq!(sccs.scc(4), 0);
    assert_eq!(sccs.successors(0), &[]);
}

#[test]
fn test_find_state_3() {
    /*
      /----+
    0 <--+ |
    |    | |
    v    | |
+-> 1 -> 3 4 5
|   |      | |
|   v      | |
+-- 2 <----+-+
     */
    let graph = TestGraph::new(0, &[
        (0, 1),
        (0, 4),
        (1, 2),
        (1, 3),
        (2, 1),
        (3, 0),
        (4, 2),
        (5, 2),
    ]);
    let sccs: Sccs<_, usize> = Sccs::new(&graph);
    assert_eq!(sccs.num_sccs(), 2);
    assert_eq!(sccs.scc(0), 0);
    assert_eq!(sccs.scc(1), 0);
    assert_eq!(sccs.scc(2), 0);
    assert_eq!(sccs.scc(3), 0);
    assert_eq!(sccs.scc(4), 0);
    assert_eq!(sccs.scc(5), 1);
    assert_eq!(sccs.successors(0), &[]);
    assert_eq!(sccs.successors(1), &[0]);
}
