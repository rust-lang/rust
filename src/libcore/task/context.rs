// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![unstable(feature = "futures_api",
            reason = "futures in libcore are unstable",
            issue = "50547")]

use fmt;
use super::{Spawn, Waker, LocalWaker};

/// Information about the currently-running task.
///
/// Contexts are always tied to the stack, since they are set up specifically
/// when performing a single `poll` step on a task.
pub struct Context<'a> {
    local_waker: &'a LocalWaker,
    spawner: &'a mut dyn Spawn,
}

impl<'a> fmt::Debug for Context<'a> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("Context")
            .finish()
    }
}

impl<'a> Context<'a> {
    /// Create a new task `Context` with the provided `local_waker`, `waker`,
    /// and `spawner`.
    #[inline]
    pub fn new(
        local_waker: &'a LocalWaker,
        spawner: &'a mut dyn Spawn,
    ) -> Context<'a> {
        Context { local_waker, spawner }
    }

    /// Get the `LocalWaker` associated with the current task.
    #[inline]
    pub fn local_waker(&self) -> &'a LocalWaker {
        self.local_waker
    }

    /// Get the `Waker` associated with the current task.
    #[inline]
    pub fn waker(&self) -> &'a Waker {
        unsafe { &*(self.local_waker as *const LocalWaker as *const Waker) }
    }

    /// Get the spawner associated with this task.
    ///
    /// This method is useful primarily if you want to explicitly handle
    /// spawn failures.
    #[inline]
    pub fn spawner(&mut self) -> &mut dyn Spawn {
        self.spawner
    }

    /// Produce a context like the current one, but using the given waker
    /// instead.
    ///
    /// This advanced method is primarily used when building "internal
    /// schedulers" within a task, where you want to provide some customized
    /// wakeup logic.
    #[inline]
    pub fn with_waker<'b>(
        &'b mut self,
        local_waker: &'b LocalWaker,
    ) -> Context<'b> {
        Context {
            local_waker,
            spawner: self.spawner,
        }
    }

    /// Produce a context like the current one, but using the given spawner
    /// instead.
    ///
    /// This advanced method is primarily used when building "internal
    /// schedulers" within a task.
    #[inline]
    pub fn with_spawner<'b, Sp: Spawn>(
        &'b mut self,
        spawner: &'b mut Sp,
    ) -> Context<'b> {
        Context {
            local_waker: self.local_waker,
            spawner,
        }
    }
}
