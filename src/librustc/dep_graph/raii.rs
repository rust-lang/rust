// Copyright 2012-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use super::DepNode;
use super::thread::{DepGraphThreadData, DepMessage};

pub struct DepTask<'graph> {
    data: &'graph DepGraphThreadData,
    key: DepNode,
}

impl<'graph> DepTask<'graph> {
    pub fn new(data: &'graph DepGraphThreadData, key: DepNode) -> DepTask<'graph> {
        data.enqueue(DepMessage::PushTask(key));
        DepTask { data: data, key: key }
    }
}

impl<'graph> Drop for DepTask<'graph> {
    fn drop(&mut self) {
        self.data.enqueue(DepMessage::PopTask(self.key));
    }
}

pub struct IgnoreTask<'graph> {
    data: &'graph DepGraphThreadData
}

impl<'graph> IgnoreTask<'graph> {
    pub fn new(data: &'graph DepGraphThreadData) -> IgnoreTask<'graph> {
        data.enqueue(DepMessage::PushIgnore);
        IgnoreTask { data: data }
    }
}

impl<'graph> Drop for IgnoreTask<'graph> {
    fn drop(&mut self) {
        self.data.enqueue(DepMessage::PopIgnore);
    }
}
