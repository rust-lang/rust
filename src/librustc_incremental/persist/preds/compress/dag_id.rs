// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use rustc_data_structures::graph::NodeIndex;
use rustc_data_structures::unify::UnifyKey;

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct DagId {
    index: u32,
}

impl DagId {
    pub fn from_input_index(n: NodeIndex) -> Self {
        DagId { index: n.0 as u32 }
    }

    pub fn as_input_index(&self) -> NodeIndex {
        NodeIndex(self.index as usize)
    }
}

impl UnifyKey for DagId {
    type Value = ();

    fn index(&self) -> u32 {
        self.index
    }

    fn from_index(u: u32) -> Self {
        DagId { index: u }
    }

    fn tag(_: Option<Self>) -> &'static str {
        "DagId"
    }
}
