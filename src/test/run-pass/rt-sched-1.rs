// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Tests of the runtime's scheduler interface

type sched_id = int;
type task_id = *libc::c_void;

type task = *libc::c_void;
type closure = *libc::c_void;

extern mod rustrt {
    #[legacy_exports];
    fn rust_new_sched(num_threads: libc::uintptr_t) -> sched_id;
    fn rust_get_sched_id() -> sched_id;
    fn rust_new_task_in_sched(id: sched_id) -> task_id;
    fn start_task(id: task_id, f: closure);
}

fn main() {
    unsafe {
        let po = oldcomm::Port();
        let ch = oldcomm::Chan(&po);
        let parent_sched_id = rustrt::rust_get_sched_id();
        error!("parent %?", parent_sched_id);
        let num_threads = 1u;
        let new_sched_id = rustrt::rust_new_sched(num_threads);
        error!("new_sched_id %?", new_sched_id);
        let new_task_id = rustrt::rust_new_task_in_sched(new_sched_id);
        assert !new_task_id.is_null();
        let f = fn~() {
            unsafe {
                let child_sched_id = rustrt::rust_get_sched_id();
                error!("child_sched_id %?", child_sched_id);
                assert child_sched_id != parent_sched_id;
                assert child_sched_id == new_sched_id;
                oldcomm::send(ch, ());
            }
        };
        let fptr = cast::reinterpret_cast(&ptr::addr_of(&f));
        rustrt::start_task(new_task_id, fptr);
        cast::forget(move f);
        oldcomm::recv(po);
    }
}
