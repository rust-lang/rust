// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::cell::{Ref, RefCell};
use rustc_data_structures::indexed_vec::IndexVec;

use mir::{Mir, BasicBlock};

use rustc_serialize as serialize;

#[derive(Clone, Debug)]
pub struct Cache {
    predecessors: RefCell<Option<IndexVec<BasicBlock, Vec<BasicBlock>>>>
}


impl serialize::Encodable for Cache {
    fn encode<S: serialize::Encoder>(&self, s: &mut S) -> Result<(), S::Error> {
        serialize::Encodable::encode(&(), s)
    }
}

impl serialize::Decodable for Cache {
    fn decode<D: serialize::Decoder>(d: &mut D) -> Result<Self, D::Error> {
        serialize::Decodable::decode(d).map(|_v: ()| Self::new())
    }
}


impl Cache {
    pub fn new() -> Self {
        Cache {
            predecessors: RefCell::new(None)
        }
    }

    pub fn invalidate(&self) {
        // FIXME: consider being more fine-grained
        *self.predecessors.borrow_mut() = None;
    }

    pub fn predecessors(&self, mir: &Mir) -> Ref<IndexVec<BasicBlock, Vec<BasicBlock>>> {
        if self.predecessors.borrow().is_none() {
            *self.predecessors.borrow_mut() = Some(calculate_predecessors(mir));
        }

        Ref::map(self.predecessors.borrow(), |p| p.as_ref().unwrap())
    }
}

fn calculate_predecessors(mir: &Mir) -> IndexVec<BasicBlock, Vec<BasicBlock>> {
    let mut result = IndexVec::from_elem(vec![], mir.basic_blocks());
    for (bb, data) in mir.basic_blocks().iter_enumerated() {
        if let Some(ref term) = data.terminator {
            for &tgt in term.successors().iter() {
                result[tgt].push(bb);
            }
        }
    }

    result
}
