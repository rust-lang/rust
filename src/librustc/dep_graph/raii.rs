// Copyright 2012-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use super::graph::CurrentDepGraph;

use std::cell::RefCell;

pub struct IgnoreTask<'graph> {
    graph: &'graph RefCell<CurrentDepGraph>,
}

impl<'graph> IgnoreTask<'graph> {
    pub(super) fn new(graph: &'graph RefCell<CurrentDepGraph>) -> IgnoreTask<'graph> {
        graph.borrow_mut().push_ignore();
        IgnoreTask {
            graph,
        }
    }
}

impl<'graph> Drop for IgnoreTask<'graph> {
    fn drop(&mut self) {
        self.graph.borrow_mut().pop_ignore();
    }
}

