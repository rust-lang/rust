use super::super::tests::TestGraph;
use super::*;

#[test]
fn diamond() {
    let graph = TestGraph::new(0, &[(0, 1), (0, 2), (1, 3), (2, 3)]);

    let d = dominators(&graph);
    assert_eq!(d.immediate_dominator(0), None);
    assert_eq!(d.immediate_dominator(1), Some(0));
    assert_eq!(d.immediate_dominator(2), Some(0));
    assert_eq!(d.immediate_dominator(3), Some(0));
}

#[test]
fn paper() {
    // example from the paper:
    let graph = TestGraph::new(
        6,
        &[(6, 5), (6, 4), (5, 1), (4, 2), (4, 3), (1, 2), (2, 3), (3, 2), (2, 1)],
    );

    let d = dominators(&graph);
    assert_eq!(d.immediate_dominator(0), None); // <-- note that 0 is not in graph
    assert_eq!(d.immediate_dominator(1), Some(6));
    assert_eq!(d.immediate_dominator(2), Some(6));
    assert_eq!(d.immediate_dominator(3), Some(6));
    assert_eq!(d.immediate_dominator(4), Some(6));
    assert_eq!(d.immediate_dominator(5), Some(6));
    assert_eq!(d.immediate_dominator(6), None);
}

#[test]
fn paper_slt() {
    // example from the paper:
    let graph = TestGraph::new(
        1,
        &[(1, 2), (1, 3), (2, 3), (2, 7), (3, 4), (3, 6), (4, 5), (5, 4), (6, 7), (7, 8), (8, 5)],
    );

    dominators(&graph);
}

#[test]
fn immediate_dominator() {
    let graph = TestGraph::new(1, &[(1, 2), (2, 3)]);
    let d = dominators(&graph);
    assert_eq!(d.immediate_dominator(0), None);
    assert_eq!(d.immediate_dominator(1), None);
    assert_eq!(d.immediate_dominator(2), Some(1));
    assert_eq!(d.immediate_dominator(3), Some(2));
}

#[test]
fn transitive_dominator() {
    let graph = TestGraph::new(
        0,
        &[
            // First tree branch.
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 4),
            // Second tree branch.
            (1, 5),
            (5, 6),
            // Third tree branch.
            (0, 7),
            // These links make 0 the dominator for 2 and 3.
            (7, 2),
            (5, 3),
        ],
    );

    let d = dominators(&graph);
    assert_eq!(d.immediate_dominator(2), Some(0));
    assert_eq!(d.immediate_dominator(3), Some(0)); // This used to return Some(1).
}
