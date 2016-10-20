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

use super::*;

#[test]
fn test1() {
    // 0 -> 1 -> 2 -> 3
    //      ^    v
    //      6 <- 4 -> 5
    let graph = TestGraph::new(0, &[(0, 1), (1, 2), (2, 3), (2, 4), (4, 5), (4, 6), (6, 1)]);
    let reachable = reachable(&graph);
    assert!((0..6).all(|i| reachable.can_reach(0, i)));
    assert!((1..6).all(|i| reachable.can_reach(1, i)));
    assert!((1..6).all(|i| reachable.can_reach(2, i)));
    assert!((1..6).all(|i| reachable.can_reach(4, i)));
    assert!((1..6).all(|i| reachable.can_reach(6, i)));
    assert!(reachable.can_reach(3, 3));
    assert!(!reachable.can_reach(3, 5));
    assert!(!reachable.can_reach(5, 3));
}

/// use bigger indices to cross between words in the bit set
#[test]
fn test2() {
    // 30 -> 31 -> 32 -> 33
    //       ^      v
    //       36 <- 34 -> 35
    let graph = TestGraph::new(30,
                               &[(30, 31), (31, 32), (32, 33), (32, 34), (34, 35), (34, 36),
                                 (36, 31)]);
    let reachable = reachable(&graph);
    assert!((30..36).all(|i| reachable.can_reach(30, i)));
    assert!((31..36).all(|i| reachable.can_reach(31, i)));
    assert!((31..36).all(|i| reachable.can_reach(32, i)));
    assert!((31..36).all(|i| reachable.can_reach(34, i)));
    assert!((31..36).all(|i| reachable.can_reach(36, i)));
    assert!(reachable.can_reach(33, 33));
    assert!(!reachable.can_reach(33, 35));
    assert!(!reachable.can_reach(35, 33));
}
