extern crate test;

use super::*;
use crate::graph::tests::TestGraph;

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
    let graph = TestGraph::new(0, &[(0, 1), (1, 2), (1, 3), (2, 0), (3, 2)]);
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
    let graph = TestGraph::new(0, &[(0, 1), (1, 2), (2, 1), (3, 2)]);
    let sccs: Sccs<_, usize> = Sccs::new(&graph);
    assert_eq!(sccs.num_sccs(), 3);
    assert_eq!(sccs.scc(0), 1);
    assert_eq!(sccs.scc(1), 0);
    assert_eq!(sccs.scc(2), 0);
    assert_eq!(sccs.scc(3), 2);
    assert_eq!(sccs.successors(0), &[] as &[usize]);
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

    let graph = TestGraph::new(0, &[(0, 1), (0, 4), (1, 2), (1, 3), (2, 1), (3, 0), (4, 2)]);

    // For this graph, we will start in our DFS by visiting:
    //
    // 0 -> 1 -> 2 -> 1
    //
    // and at this point detect a cycle. The state of 2 will thus be
    // `InCycleWith { 1 }`. We will then visit the 1 -> 3 edge, which
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
    assert_eq!(sccs.successors(0), &[] as &[usize]);
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
    let graph =
        TestGraph::new(0, &[(0, 1), (0, 4), (1, 2), (1, 3), (2, 1), (3, 0), (4, 2), (5, 2)]);
    let sccs: Sccs<_, usize> = Sccs::new(&graph);
    assert_eq!(sccs.num_sccs(), 2);
    assert_eq!(sccs.scc(0), 0);
    assert_eq!(sccs.scc(1), 0);
    assert_eq!(sccs.scc(2), 0);
    assert_eq!(sccs.scc(3), 0);
    assert_eq!(sccs.scc(4), 0);
    assert_eq!(sccs.scc(5), 1);
    assert_eq!(sccs.successors(0), &[] as &[usize]);
    assert_eq!(sccs.successors(1), &[0]);
}

#[test]
fn test_deep_linear() {
    /*
    0
    |
    v
    1
    |
    v
    2
    |
    v
    …
     */
    #[cfg(not(miri))]
    const NR_NODES: usize = 1 << 14;
    #[cfg(miri)]
    const NR_NODES: usize = 1 << 3;
    let mut nodes = vec![];
    for i in 1..NR_NODES {
        nodes.push((i - 1, i));
    }
    let graph = TestGraph::new(0, nodes.as_slice());
    let sccs: Sccs<_, usize> = Sccs::new(&graph);
    assert_eq!(sccs.num_sccs(), NR_NODES);
    assert_eq!(sccs.scc(0), NR_NODES - 1);
    assert_eq!(sccs.scc(NR_NODES - 1), 0);
}

#[bench]
fn bench_sccc(b: &mut test::Bencher) {
    // Like `test_three_sccs` but each state is replaced by a group of
    // three or four to have some amount of test data.
    /*
       0-3
        |
        v
    +->4-6 11-14
    |   |    |
    |   v    |
    +--7-10<-+
         */
    fn make_3_clique(slice: &mut [(usize, usize)], base: usize) {
        slice[0] = (base + 0, base + 1);
        slice[1] = (base + 1, base + 2);
        slice[2] = (base + 2, base + 0);
    }
    // Not actually a clique but strongly connected.
    fn make_4_clique(slice: &mut [(usize, usize)], base: usize) {
        slice[0] = (base + 0, base + 1);
        slice[1] = (base + 1, base + 2);
        slice[2] = (base + 2, base + 3);
        slice[3] = (base + 3, base + 0);
        slice[4] = (base + 1, base + 3);
        slice[5] = (base + 2, base + 1);
    }

    let mut graph = [(0, 0); 6 + 3 + 6 + 3 + 4];
    make_4_clique(&mut graph[0..6], 0);
    make_3_clique(&mut graph[6..9], 4);
    make_4_clique(&mut graph[9..15], 7);
    make_3_clique(&mut graph[15..18], 11);
    graph[18] = (0, 4);
    graph[19] = (5, 7);
    graph[20] = (11, 10);
    graph[21] = (7, 4);
    let graph = TestGraph::new(0, &graph[..]);
    b.iter(|| {
        let sccs: Sccs<_, usize> = Sccs::new(&graph);
        assert_eq!(sccs.num_sccs(), 3);
    });
}
