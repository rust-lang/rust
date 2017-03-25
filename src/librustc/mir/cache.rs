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

use mir::{Mir, Block};

use rustc_serialize as serialize;

#[derive(Clone, Debug)]
pub struct Cache {
    predecessors: RefCell<Option<IndexVec<Block, Vec<Block>>>>,
    successors: RefCell<Option<IndexVec<Block, Vec<Block>>>>,
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
            predecessors: RefCell::new(None),
            successors: RefCell::new(None)
        }
    }

    pub fn invalidate(&self) {
        // FIXME: consider being more fine-grained
        *self.predecessors.borrow_mut() = None;
        *self.successors.borrow_mut() = None;
    }

    pub fn predecessors(&self, mir: &Mir) -> Ref<IndexVec<Block, Vec<Block>>> {
        if self.predecessors.borrow().is_none() {
            *self.predecessors.borrow_mut() = Some(self.calculate_predecessors(mir));
        }

        Ref::map(self.predecessors.borrow(), |p| p.as_ref().unwrap())
    }

    fn calculate_predecessors(&self, mir: &Mir) -> IndexVec<Block, Vec<Block>> {
        let mut result = IndexVec::from_elem(vec![], mir.basic_blocks());
        for (bb, bbs) in self.successors(mir).iter_enumerated() {
            for &tgt in bbs {
                result[tgt].push(bb);
            }
        }

        result
    }

    pub fn successors(&self, mir: &Mir) -> Ref<IndexVec<Block, Vec<Block>>> {
        if self.successors.borrow().is_none() {
            *self.successors.borrow_mut() = Some(calculate_successors(mir));
        }

        Ref::map(self.successors.borrow(), |p| p.as_ref().unwrap())
    }
}

fn calculate_successors(mir: &Mir) -> IndexVec<Block, Vec<Block>> {
    let mut result = IndexVec::from_elem(vec![], mir.basic_blocks());
    for (bb, data) in mir.basic_blocks().iter_enumerated() {
        for stmt in &data.statements {
            if let Some(cleanup) = stmt.cleanup_target() {
                result[bb].push(cleanup);
            }
        }

        if let Some(ref term) = data.terminator {
            for &tgt in term.successors().iter() {
                result[bb].push(tgt);
            }
        }
    }

    result
}
