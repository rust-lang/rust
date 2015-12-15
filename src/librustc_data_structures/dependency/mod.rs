// Copyright 2012-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Dependency graph used for incremental compilation.

use std::cell::RefCell;
use std::fmt::Debug;
use std::hash::Hash;
use self::state::DepGraphState;

mod state;

#[cfg(test)]
mod test;

pub struct DepGraph<ID: DepNodeId> {
    state: RefCell<DepGraphState<ID>>
}

pub trait DepNodeId: Clone + Debug + Hash + PartialEq + Eq {
}

impl<ID: DepNodeId> DepGraph<ID> {
    pub fn new() -> DepGraph<ID> {
        DepGraph {
            state: RefCell::new(DepGraphState::new())
        }
    }

    pub fn dependents(&self, node: ID) -> Vec<ID> {
        self.state.borrow().dependents(node)
    }

    pub fn in_ignore<'graph>(&'graph self) -> IgnoreTask<'graph, ID> {
        IgnoreTask::new(self)
    }

    pub fn with_ignore<OP,R>(&self, op: OP) -> R
        where OP: FnOnce() -> R
    {
        let _task = self.in_ignore();
        op()
    }

    pub fn in_task<'graph>(&'graph self, key: ID) -> DepTask<'graph, ID> {
        DepTask::new(self, key)
    }

    pub fn with_task<OP,R>(&self, key: ID, op: OP) -> R
        where OP: FnOnce() -> R
    {
        let _task = self.in_task(key);
        op()
    }

    /// Indicates that the current task `C` reads `v` by adding an
    /// edge from `v` to `C`. If there is no current task, panics. If
    /// you want to suppress this edge, use `ignore`.
    pub fn read(&self, v: ID) {
        self.state.borrow_mut().read(v);
    }

    /// Indicates that the current task `C` writes `v` by adding an
    /// edge from `C` to `v`. If there is no current task, panics. If
    /// you want to suppress this edge, use `ignore`.
    pub fn write(&self, v: ID) {
        self.state.borrow_mut().write(v);
    }

    // Low-level graph introspection, mainly intended for dumping out state.

    pub fn nodes(&self) -> Vec<ID> {
        self.state.borrow().nodes()
    }

    pub fn edges(&self) -> Vec<(ID,ID)> {
        self.state.borrow().edges()
    }
}

pub struct DepTask<'graph, ID: 'graph + DepNodeId> {
    graph: &'graph DepGraph<ID>,
    key: ID,
}

impl<'graph, ID: DepNodeId> DepTask<'graph, ID> {
    pub fn new(graph: &'graph DepGraph<ID>, key: ID) -> DepTask<'graph, ID> {
        graph.state.borrow_mut().push_task(key.clone());
        DepTask { graph: graph, key: key }
    }
}

impl<'graph, ID: DepNodeId> Drop for DepTask<'graph, ID> {
    fn drop(&mut self) {
        self.graph.state.borrow_mut().pop_task(self.key.clone());
    }
}

pub struct IgnoreTask<'graph, ID: 'graph + DepNodeId> {
    graph: &'graph DepGraph<ID>,
}

impl<'graph, ID: DepNodeId> IgnoreTask<'graph, ID> {
    pub fn new(graph: &'graph DepGraph<ID>) -> IgnoreTask<'graph, ID> {
        graph.state.borrow_mut().push_ignore();
        IgnoreTask { graph: graph }
    }
}

impl<'graph, ID: DepNodeId> Drop for IgnoreTask<'graph, ID> {
    fn drop(&mut self) {
        self.graph.state.borrow_mut().pop_ignore();
    }
}
