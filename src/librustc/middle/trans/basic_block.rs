// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use llvm;
use llvm::{BasicBlockRef};
use middle::trans::value::{Users, Value};
use std::iter::{Filter, Map};

pub struct BasicBlock(pub BasicBlockRef);

pub type Preds<'a> = Map<'a, Value, BasicBlock, Filter<'a, Value, Users>>;

/**
 * Wrapper for LLVM BasicBlockRef
 */
impl BasicBlock {
    pub fn get(&self) -> BasicBlockRef {
        let BasicBlock(v) = *self; v
    }

    pub fn as_value(self) -> Value {
        unsafe {
            Value(llvm::LLVMBasicBlockAsValue(self.get()))
        }
    }

    pub fn pred_iter(self) -> Preds<'static> {
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
