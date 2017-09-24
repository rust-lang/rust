// Copyright 2012-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use super::edges::DepGraphEdges;
use super::graph::CurrentDepGraph;

use std::cell::RefCell;

pub struct IgnoreTask<'graph> {
    legacy_graph: &'graph RefCell<DepGraphEdges>,
    new_graph: &'graph RefCell<CurrentDepGraph>,
}

impl<'graph> IgnoreTask<'graph> {
    pub(super) fn new(legacy_graph: &'graph RefCell<DepGraphEdges>,
                      new_graph: &'graph RefCell<CurrentDepGraph>)
                      -> IgnoreTask<'graph> {
        legacy_graph.borrow_mut().push_ignore();
        new_graph.borrow_mut().push_ignore();
        IgnoreTask {
            legacy_graph,
            new_graph,
        }
    }
}

impl<'graph> Drop for IgnoreTask<'graph> {
    fn drop(&mut self) {
        self.legacy_graph.borrow_mut().pop_ignore();
        self.new_graph.borrow_mut().pop_ignore();
    }
}

