// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/*!

The task interface to the runtime

*/

#[doc(hidden)];

use libc;

#[allow(non_camel_case_types)] // runtime type
pub type sched_id = int;
#[allow(non_camel_case_types)] // runtime type
pub type task_id = int;

// These are both opaque runtime/compiler types that we don't know the
// structure of and should only deal with via unsafe pointer
#[allow(non_camel_case_types)] // runtime type
pub type rust_task = libc::c_void;
#[allow(non_camel_case_types)] // runtime type
pub type rust_closure = libc::c_void;

extern {
    #[rust_stack]
    pub fn rust_task_yield(task: *rust_task) -> bool;

    pub fn rust_get_sched_id() -> sched_id;
    pub fn rust_new_sched(num_threads: libc::uintptr_t) -> sched_id;

    pub fn get_task_id() -> task_id;
    #[rust_stack]
    pub fn rust_get_task() -> *rust_task;

    pub fn new_task() -> *rust_task;
    pub fn rust_new_task_in_sched(id: sched_id) -> *rust_task;

    pub fn start_task(task: *rust_task, closure: *rust_closure);

    pub fn rust_task_is_unwinding(task: *rust_task) -> bool;
    pub fn rust_osmain_sched_id() -> sched_id;
    #[rust_stack]
    pub fn rust_task_inhibit_kill(t: *rust_task);
    #[rust_stack]
    pub fn rust_task_allow_kill(t: *rust_task);
    #[rust_stack]
    pub fn rust_task_inhibit_yield(t: *rust_task);
    #[rust_stack]
    pub fn rust_task_allow_yield(t: *rust_task);
    pub fn rust_task_kill_other(task: *rust_task);
    pub fn rust_task_kill_all(task: *rust_task);

    #[rust_stack]
    pub fn rust_get_task_local_data(task: *rust_task) -> *mut *libc::c_void;
    #[rust_stack]
    pub fn rust_task_local_data_atexit(task: *rust_task, cleanup_fn: *u8);
}
