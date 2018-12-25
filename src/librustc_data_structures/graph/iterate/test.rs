use super::super::test::TestGraph;

use super::*;

#[test]
fn diamond_post_order() {
    let graph = TestGraph::new(0, &[(0, 1), (0, 2), (1, 3), (2, 3)]);

    let result = post_order_from(&graph, 0);
    assert_eq!(result, vec![3, 1, 2, 0]);
}
