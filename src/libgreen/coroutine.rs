// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Coroutines represent nothing more than a context and a stack
// segment.

use context::Context;
use stack::{StackPool, Stack};

/// A coroutine is nothing more than a (register context, stack) pair.
pub struct Coroutine {
    /// The segment of stack on which the task is currently running or
    /// if the task is blocked, on which the task will resume
    /// execution.
    ///
    /// Servo needs this to be public in order to tell SpiderMonkey
    /// about the stack bounds.
    pub current_stack_segment: Stack,

    /// Always valid if the task is alive and not running.
    pub saved_context: Context
}

impl Coroutine {
    pub fn empty() -> Coroutine {
        Coroutine {
            current_stack_segment: unsafe { Stack::dummy_stack() },
            saved_context: Context::empty()
        }
    }

    /// Destroy coroutine and try to reuse std::stack segment.
    pub fn recycle(self, stack_pool: &mut StackPool) {
        let Coroutine { current_stack_segment, .. } = self;
        stack_pool.give_stack(current_stack_segment);
    }
}
