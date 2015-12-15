// Copyright 2012-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use super::{DepGraph, DepNodeId};

impl DepNodeId for &'static str { }

#[test]
fn foo() {
    let graph = DepGraph::new();

    graph.with_task("a", || {
        graph.read("b");
        graph.write("c");
    });

    let mut deps = graph.dependents("a");
    deps.sort();
    assert_eq!(deps, vec!["a", "c"]);

    let mut deps = graph.dependents("b");
    deps.sort();
    assert_eq!(deps, vec!["a", "b", "c"]);

    let mut deps = graph.dependents("c");
    deps.sort();
    assert_eq!(deps, vec!["c"]);
}
