// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Language-level runtime services that should reasonably expected
//! to be available 'everywhere'. Local heaps, GC, unwinding,
//! local storage, and logging. Even a 'freestanding' Rust would likely want
//! to implement this.

//! Local services may exist in at least three different contexts:
//! when running as a task, when running in the scheduler's context,
//! or when running outside of a scheduler but with local services
//! (freestanding rust with local services?).

use prelude::*;
use super::sched::{Task, local_sched};

pub struct LocalServices {
    heap: LocalHeap,
    gc: GarbageCollector,
    storage: LocalStorage,
    logger: Logger,
    unwinder: Unwinder
}

pub struct LocalHeap;
pub struct GarbageCollector;
pub struct LocalStorage;
pub struct Logger;
pub struct Unwinder;

impl LocalServices {
    pub fn new() -> LocalServices {
        LocalServices {
            heap: LocalHeap,
            gc: GarbageCollector,
            storage: LocalStorage,
            logger: Logger,
            unwinder: Unwinder
        }
    }
}

/// Borrow a pointer to the installed local services.
/// Fails (likely aborting the process) if local services are not available.
pub fn borrow_local_services(f: &fn(&mut LocalServices)) {
    do local_sched::borrow |sched| {
        match sched.current_task {
            Some(~ref mut task) => {
                f(&mut task.local_services)
            }
            None => {
                fail!(~"no local services for schedulers yet")
            }
        }
    }
}
