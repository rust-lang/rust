// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use super::super::test::TestGraph;
use super::super::transpose::TransposedGraph;

use super::*;

#[test]
fn diamond_post_order() {
    let graph = TestGraph::new(0, &[(0, 1), (0, 2), (1, 3), (2, 3)]);

    let result = post_order_from(&graph, 0);
    assert_eq!(result, vec![3, 1, 2, 0]);
}


#[test]
fn rev_post_order_inner_loop() {
    // 0 -> 1 ->     2     -> 3 -> 5
    //      ^     ^    v      |
    //      |     6 <- 4      |
    //      +-----------------+
    let graph = TestGraph::new(0,
                               &[(0, 1), (1, 2), (2, 3), (3, 5), (3, 1), (2, 4), (4, 6), (6, 2)]);

    let rev_graph = TransposedGraph::new(&graph);

    let result = post_order_from_to(&rev_graph, 6, Some(2));
    assert_eq!(result, vec![4, 6]);

    let result = post_order_from_to(&rev_graph, 3, Some(1));
    assert_eq!(result, vec![4, 6, 2, 3]);
}
