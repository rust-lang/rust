// Copyright 2012-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use hir::def_id::DefId;
use super::DepNode;
use super::thread::{DepGraphThreadData, DepMessage};

pub struct DepTask<'graph> {
    data: &'graph DepGraphThreadData,
    key: Option<DepNode<DefId>>,
}

impl<'graph> DepTask<'graph> {
    pub fn new(data: &'graph DepGraphThreadData, key: DepNode<DefId>)
               -> Option<DepTask<'graph>> {
        if data.is_enqueue_enabled() {
            data.enqueue(DepMessage::PushTask(key.clone()));
            Some(DepTask { data: data, key: Some(key) })
        } else {
            None
        }
    }
}

impl<'graph> Drop for DepTask<'graph> {
    fn drop(&mut self) {
        if self.data.is_enqueue_enabled() {
            self.data.enqueue(DepMessage::PopTask(self.key.take().unwrap()));
        }
    }
}

pub struct IgnoreTask<'graph> {
    data: &'graph DepGraphThreadData
}

impl<'graph> IgnoreTask<'graph> {
    pub fn new(data: &'graph DepGraphThreadData) -> Option<IgnoreTask<'graph>> {
        if data.is_enqueue_enabled() {
            data.enqueue(DepMessage::PushIgnore);
            Some(IgnoreTask { data: data })
        } else {
            None
        }
    }
}

impl<'graph> Drop for IgnoreTask<'graph> {
    fn drop(&mut self) {
        if self.data.is_enqueue_enabled() {
            self.data.enqueue(DepMessage::PopIgnore);
        }
    }
}

