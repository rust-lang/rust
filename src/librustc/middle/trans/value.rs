// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use lib::llvm::{llvm, UseRef, ValueRef};
use middle::trans::basic_block::BasicBlock;
use middle::trans::common::Block;
use libc::c_uint;

pub struct Value(ValueRef);

macro_rules! opt_val ( ($e:expr) => (
    unsafe {
        match $e {
            p if p.is_not_null() => Some(Value(p)),
            _ => None
        }
    }
))

/**
 * Wrapper for LLVM ValueRef
 */
impl Value {
    /// Returns the native ValueRef
    pub fn get(&self) -> ValueRef {
        let Value(v) = *self; v
    }

    /// Returns the BasicBlock that contains this value
    pub fn get_parent(self) -> Option<BasicBlock> {
        unsafe {
            match llvm::LLVMGetInstructionParent(self.get()) {
                p if p.is_not_null() => Some(BasicBlock(p)),
                _ => None
            }
        }
    }

    /// Removes this value from its containing BasicBlock
    pub fn erase_from_parent(self) {
        unsafe {
            llvm::LLVMInstructionEraseFromParent(self.get());
        }
    }

    /// Returns the single dominating store to this value, if any
    /// This only performs a search for a trivially dominating store. The store
    /// must be the only user of this value, and there must not be any conditional
    /// branches between the store and the given block.
    pub fn get_dominating_store(self, bcx: &Block) -> Option<Value> {
        match self.get_single_user().and_then(|user| user.as_store_inst()) {
            Some(store) => {
                store.get_parent().and_then(|store_bb| {
                    let mut bb = BasicBlock(bcx.llbb);
                    let mut ret = Some(store);
                    while bb.get() != store_bb.get() {
                        match bb.get_single_predecessor() {
                            Some(pred) => bb = pred,
                            None => { ret = None; break }
                        }
                    }
                    ret
                })
            }
            _ => None
        }
    }

    /// Returns the first use of this value, if any
    pub fn get_first_use(self) -> Option<Use> {
        unsafe {
            match llvm::LLVMGetFirstUse(self.get()) {
                u if u.is_not_null() => Some(Use(u)),
                _ => None
            }
        }
    }

    /// Tests if there are no uses of this value
    pub fn has_no_uses(self) -> bool {
        self.get_first_use().is_none()
    }

    /// Returns the single user of this value
    /// If there are no users or multiple users, this returns None
    pub fn get_single_user(self) -> Option<Value> {
        let mut iter = self.user_iter();
        match (iter.next(), iter.next()) {
            (Some(first), None) => Some(first),
            _ => None
        }
    }

    /// Returns an iterator for the users of this value
    pub fn user_iter(self) -> Users {
        Users {
            next: self.get_first_use()
        }
    }

    /// Returns the requested operand of this instruction
    /// Returns None, if there's no operand at the given index
    pub fn get_operand(self, i: uint) -> Option<Value> {
        opt_val!(llvm::LLVMGetOperand(self.get(), i as c_uint))
    }

    /// Returns the Store represent by this value, if any
    pub fn as_store_inst(self) -> Option<Value> {
        opt_val!(llvm::LLVMIsAStoreInst(self.get()))
    }

    /// Tests if this value is a terminator instruction
    pub fn is_a_terminator_inst(self) -> bool {
        unsafe {
            llvm::LLVMIsATerminatorInst(self.get()).is_not_null()
        }
    }
}

pub struct Use(UseRef);

/**
 * Wrapper for LLVM UseRef
 */
impl Use {
    pub fn get(&self) -> UseRef {
        let Use(v) = *self; v
    }

    pub fn get_user(self) -> Value {
        unsafe {
            Value(llvm::LLVMGetUser(self.get()))
        }
    }

    pub fn get_next_use(self) -> Option<Use> {
        unsafe {
            match llvm::LLVMGetNextUse(self.get()) {
                u if u.is_not_null() => Some(Use(u)),
                _ => None
            }
        }
    }
}

/// Iterator for the users of a value
pub struct Users {
    priv next: Option<Use>
}

impl Iterator<Value> for Users {
    fn next(&mut self) -> Option<Value> {
        let current = self.next;

        self.next = current.and_then(|u| u.get_next_use());

        current.map(|u| u.get_user())
    }
}
