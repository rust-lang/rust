// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use lib::llvm::{llvm, BasicBlockRef};
use middle::trans::value::{UserIterator, Value};
use std::iter::{Filter, Map};

pub struct BasicBlock(BasicBlockRef);

pub type PredIterator<'a> = Map<'a, Value, BasicBlock, Filter<'a, Value, UserIterator>>;

/**
 * Wrapper for LLVM BasicBlockRef
 */
impl BasicBlock {
    pub fn as_value(self) -> Value {
        unsafe {
            Value(llvm::LLVMBasicBlockAsValue(*self))
        }
    }

    pub fn pred_iter(self) -> PredIterator {
        self.as_value().user_iter()
            .filter(|user| user.is_a_terminator_inst())
            .map(|user| user.get_parent().unwrap())
    }

    pub fn get_single_predecessor(self) -> Option<BasicBlock> {
        let mut iter = self.pred_iter();
        match (iter.next(), iter.next()) {
            (Some(first), None) => Some(first),
            _ => None
        }
    }
}
