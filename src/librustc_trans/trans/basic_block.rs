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
use trans::value::{Users, Value};
use std::iter::{Filter, Map};

#[deriving(Copy)]
pub struct BasicBlock(pub BasicBlockRef);

pub type Preds = Map<
    Value,
    BasicBlock,
    Filter<Value, Users, fn(&Value) -> bool>,
    fn(Value) -> BasicBlock,
>;

/// Wrapper for LLVM BasicBlockRef
impl BasicBlock {
    pub fn get(&self) -> BasicBlockRef {
        let BasicBlock(v) = *self; v
    }

    pub fn as_value(self) -> Value {
        unsafe {
            Value(llvm::LLVMBasicBlockAsValue(self.get()))
        }
    }

    pub fn pred_iter(self) -> Preds {
        fn is_a_terminator_inst(user: &Value) -> bool { user.is_a_terminator_inst() }
        let is_a_terminator_inst: fn(&Value) -> bool = is_a_terminator_inst;

        fn get_parent(user: Value) -> BasicBlock { user.get_parent().unwrap() }
        let get_parent: fn(Value) -> BasicBlock = get_parent;

        self.as_value().user_iter()
            .filter(is_a_terminator_inst)
            .map(get_parent)
    }

    pub fn get_single_predecessor(self) -> Option<BasicBlock> {
        let mut iter = self.pred_iter();
        match (iter.next(), iter.next()) {
            (Some(first), None) => Some(first),
            _ => None
        }
    }
}
