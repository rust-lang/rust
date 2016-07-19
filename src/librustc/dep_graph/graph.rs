// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use hir::def_id::DefId;
use std::rc::Rc;

use super::dep_node::DepNode;
use super::query::DepGraphQuery;
use super::raii;
use super::thread::{DepGraphThreadData, DepMessage};

#[derive(Clone)]
pub struct DepGraph {
    data: Rc<DepGraphThreadData>
}

impl DepGraph {
    pub fn new(enabled: bool) -> DepGraph {
        DepGraph {
            data: Rc::new(DepGraphThreadData::new(enabled))
        }
    }

    /// True if we are actually building a dep-graph. If this returns false,
    /// then the other methods on this `DepGraph` will have no net effect.
    #[inline]
    pub fn enabled(&self) -> bool {
        self.data.enabled()
    }

    pub fn query(&self) -> DepGraphQuery<DefId> {
        self.data.query()
    }

    pub fn in_ignore<'graph>(&'graph self) -> raii::IgnoreTask<'graph> {
        raii::IgnoreTask::new(&self.data)
    }

    pub fn in_task<'graph>(&'graph self, key: DepNode<DefId>) -> raii::DepTask<'graph> {
        raii::DepTask::new(&self.data, key)
    }

    pub fn with_ignore<OP,R>(&self, op: OP) -> R
        where OP: FnOnce() -> R
    {
        let _task = self.in_ignore();
        op()
    }

    pub fn with_task<OP,R>(&self, key: DepNode<DefId>, op: OP) -> R
        where OP: FnOnce() -> R
    {
        let _task = self.in_task(key);
        op()
    }

    pub fn read(&self, v: DepNode<DefId>) {
        self.data.enqueue(DepMessage::Read(v));
    }

    pub fn write(&self, v: DepNode<DefId>) {
        self.data.enqueue(DepMessage::Write(v));
    }
}
