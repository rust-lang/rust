use super::*;

fn reduce(graph: &Graph<&'static str, ()>,
          inputs: &[&'static str],
          outputs: &[&'static str],
          expected: &[&'static str])
{
    let reduce = GraphReduce::new(&graph,
                                  |n| inputs.contains(n),
                                  |n| outputs.contains(n));
    let result = reduce.compute();
    let mut edges: Vec<String> =
        result.graph
              .all_edges()
              .iter()
              .map(|edge| format!("{} -> {}",
                                  result.graph.node_data(edge.source()),
                                  result.graph.node_data(edge.target())))
              .collect();
    edges.sort();
    println!("{:#?}", edges);
    assert_eq!(edges.len(), expected.len());
    for (expected, actual) in expected.iter().zip(&edges) {
        assert_eq!(expected, actual);
    }
}

#[test]
fn test1() {
    //  +---------------+
    //  |               |
    //  |      +--------|------+
    //  |      |        v      v
    // [A] -> [C0] -> [C1]    [D]
    //        [  ] <- [  ] -> [E]
    //                  ^
    // [B] -------------+
    let (graph, _nodes) = graph! {
        A -> C0,
        A -> C1,
        B -> C1,
        C0 -> C1,
        C1 -> C0,
        C0 -> D,
        C1 -> E,
    };

    // [A] -> [C1] -> [D]
    // [B] -> [  ] -> [E]
    reduce(&graph, &["A", "B"], &["D", "E"], &[
        "A -> C1",
        "B -> C1",
        "C1 -> D",
        "C1 -> E",
    ]);
}

#[test]
fn test2() {
    //  +---------------+
    //  |               |
    //  |      +--------|------+
    //  |      |        v      v
    // [A] -> [C0] -> [C1]    [D] -> [E]
    //        [  ] <- [  ]
    //                  ^
    // [B] -------------+
    let (graph, _nodes) = graph! {
        A -> C0,
        A -> C1,
        B -> C1,
        C0 -> C1,
        C1 -> C0,
        C0 -> D,
        D -> E,
    };

    // [A] -> [D] -> [E]
    // [B] -> [ ]
    reduce(&graph, &["A", "B"], &["D", "E"], &[
        "A -> D",
        "B -> D",
        "D -> E",
    ]);
}

#[test]
fn test2b() {
    // Variant on test2 in which [B] is not
    // considered an input.
    let (graph, _nodes) = graph! {
        A -> C0,
        A -> C1,
        B -> C1,
        C0 -> C1,
        C1 -> C0,
        C0 -> D,
        D -> E,
    };

    // [A] -> [D] -> [E]
    reduce(&graph, &["A"], &["D", "E"], &[
        "A -> D",
        "D -> E",
    ]);
}

#[test]
fn test3() {

    // Edges going *downwards*, so 0, 1 and 2 are inputs,
    // while 7, 8, and 9 are outputs.
    //
    //     0     1   2
    //     |      \ /
    //     3---+   |
    //     |   |   |
    //     |   |   |
    //     4   5   6
    //      \ / \ / \
    //       |   |   |
    //       7   8   9
    //
    // Here the end result removes node 4, instead encoding an edge
    // from n3 -> n7, but keeps nodes 5 and 6, as they are common
    // inputs to nodes 8/9.

    let (graph, _nodes) = graph! {
        n0 -> n3,
        n3 -> n4,
        n3 -> n5,
        n4 -> n7,
        n5 -> n7,
        n5 -> n8,
        n1 -> n6,
        n2 -> n6,
        n6 -> n8,
        n6 -> n9,
    };

    reduce(&graph, &["n0", "n1", "n2"], &["n7", "n8", "n9"], &[
        "n0 -> n3",
        "n1 -> n6",
        "n2 -> n6",
        "n3 -> n5",
        "n3 -> n7",
        "n5 -> n7",
        "n5 -> n8",
        "n6 -> n8",
        "n6 -> n9"
    ]);
}

//#[test]
//fn test_cached_dfs_cyclic() {
//
//    //    0       1 <---- 2       3
//    //    ^       |       ^       ^
//    //    |       v       |       |
//    //    4 ----> 5 ----> 6 ----> 7
//    //    ^       ^       ^       ^
//    //    |       |       |       |
//    //    8       9      10      11
//
//
//    let mut g: Graph<bool, ()> = Graph::new();
//    g.add_node(false);
//    g.add_node(false);
//    g.add_node(false);
//    g.add_node(false);
//    g.add_node(false);
//    g.add_node(false);
//    g.add_node(false);
//    g.add_node(false);
//    g.add_node(true);
//    g.add_node(true);
//    g.add_node(true);
//    g.add_node(true);
//
//    g.add_edge(NodeIndex( 4), NodeIndex(0), ());
//    g.add_edge(NodeIndex( 8), NodeIndex(4), ());
//    g.add_edge(NodeIndex( 4), NodeIndex(5), ());
//    g.add_edge(NodeIndex( 1), NodeIndex(5), ());
//    g.add_edge(NodeIndex( 9), NodeIndex(5), ());
//    g.add_edge(NodeIndex( 5), NodeIndex(6), ());
//    g.add_edge(NodeIndex( 6), NodeIndex(2), ());
//    g.add_edge(NodeIndex( 2), NodeIndex(1), ());
//    g.add_edge(NodeIndex(10), NodeIndex(6), ());
//    g.add_edge(NodeIndex( 6), NodeIndex(7), ());
//    g.add_edge(NodeIndex(11), NodeIndex(7), ());
//    g.add_edge(NodeIndex( 7), NodeIndex(3), ());
//
//    let mut ws1 = DfsWorkspace::new(g.len_nodes());
//    let mut ws2 = DfsWorkspace::new(g.len_nodes());
//    let mut visit_counts: Vec<_> = g.all_nodes().iter().map(|_| 0u32).collect();
//    let mut cache: Vec<Option<Box<[u32]>>> = g.all_nodes().iter().map(|_| None).collect();
//
//    fn is_root(x: &bool) -> bool { *x }
//
//    for _ in 0 .. CACHING_THRESHOLD + 1 {
//        find_roots(&g, 2, &mut visit_counts, &mut cache[..], is_root, &mut ws1, Some(&mut ws2));
//        ws1.output.nodes.sort();
//        assert_eq!(ws1.output.nodes, vec![8, 9, 10]);
//
//        find_roots(&g, 3, &mut visit_counts, &mut cache[..], is_root, &mut ws1, Some(&mut ws2));
//        ws1.output.nodes.sort();
//        assert_eq!(ws1.output.nodes, vec![8, 9, 10, 11]);
//    }
//}

#[test]
fn test_cycle_output() {
    //  +---------------+
    //  |               |
    //  |      +--------|------+
    //  |      |        v      v
    // [A] -> [C0] <-> [C1] <- [D]
    //                  +----> [E]
    //                          ^
    // [B] ----------------- ---+
    let (graph, _nodes) = graph! {
        A -> C0,
        A -> C1,
        B -> E,
        C0 -> C1,
        C1 -> C0,
        C0 -> D,
        C1 -> E,
        D -> C1,
    };

    // [A] -> [C0] <-> [D]
    //          +----> [E]
    //                  ^
    // [B] -------------+
    reduce(&graph, &["A", "B"], &["D", "E"], &[
        "A -> C0",
        "B -> E",
        "C0 -> D",
        "C0 -> E",
        "D -> C0"
    ]);
}
