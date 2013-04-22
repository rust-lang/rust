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
use libc::c_void;
use super::sched::{Task, local_sched};
use super::local_heap::LocalHeap;

pub struct LocalServices {
    heap: LocalHeap,
    gc: GarbageCollector,
    storage: LocalStorage,
    logger: Logger,
    unwinder: Unwinder,
    destroyed: bool
}

pub struct GarbageCollector;
pub struct LocalStorage(*c_void, Option<~fn(*c_void)>);
pub struct Logger;
pub struct Unwinder;

impl LocalServices {
    pub fn new() -> LocalServices {
        LocalServices {
            heap: LocalHeap::new(),
            gc: GarbageCollector,
            storage: LocalStorage(ptr::null(), None),
            logger: Logger,
            unwinder: Unwinder,
            destroyed: false
        }
    }

    /// Must be called manually before finalization to clean up
    /// thread-local resources. Some of the routines here expect
    /// LocalServices to be available recursively so this must be
    /// called unsafely, without removing LocalServices from
    /// thread-local-storage.
    pub fn destroy(&mut self) {
        // This is just an assertion that `destroy` was called unsafely
        // and this instance of LocalServices is still accessible.
        do borrow_local_services |sched| {
            assert!(ptr::ref_eq(sched, self));
        }
        match self.storage {
            LocalStorage(ptr, Some(ref dtor)) => (*dtor)(ptr),
            _ => ()
        }
        self.destroyed = true;
    }
}

impl Drop for LocalServices {
    fn finalize(&self) { assert!(self.destroyed) }
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

pub unsafe fn unsafe_borrow_local_services() -> &mut LocalServices {
    use cast::transmute_mut_region;

    match local_sched::unsafe_borrow().current_task {
        Some(~ref mut task) => {
            transmute_mut_region(&mut task.local_services)
        }
        None => {
            fail!(~"no local services for schedulers yet")
        }
    }
}

#[cfg(test)]
mod test {
    use rt::test::*;

    #[test]
    fn local_heap() {
        do run_in_newsched_task() {
            let a = @5;
            let b = a;
            assert!(*a == 5);
            assert!(*b == 5);
        }
    }

    #[test]
    fn tls() {
        use task::local_data::*;
        do run_in_newsched_task() {
            unsafe {
                fn key(_x: @~str) { }
                local_data_set(key, @~"data");
                assert!(*local_data_get(key).get() == ~"data");
            }
        }
    }
}