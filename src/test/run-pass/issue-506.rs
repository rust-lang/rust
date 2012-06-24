/*
  A reduced test case for Issue #506, provided by Rob Arnold.
*/

use std;
import task;

#[abi = "cdecl"]
native mod rustrt {
    fn rust_task_allow_kill();
}

fn main() { task::spawn(rustrt::rust_task_allow_kill); }
