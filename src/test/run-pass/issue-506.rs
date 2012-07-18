/*
  A reduced test case for Issue #506, provided by Rob Arnold.

  Testing spawning foreign functions
*/

use std;
import task;

#[abi = "cdecl"]
extern mod rustrt {
    fn get_task_id() -> libc::intptr_t;
}

fn main() {
    let f: fn() -> libc::intptr_t = rustrt::get_task_id;
    task::spawn(unsafe { unsafe::reinterpret_cast(f) });
}
