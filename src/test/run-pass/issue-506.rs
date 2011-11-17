// xfail-test
// FIXME: This test is no longer testing what it was intended to. It should
// be testing spawning of a native function, but is actually testing
// spawning some other function, then executing a native function.

/*
  A reduced test case for Issue #506, provided by Rob Arnold.
*/

use std;
import std::task;

#[abi = "cdecl"]
native mod rustrt {
    fn task_yield();
}

fn yield_wrap(&&_arg: ()) { rustrt::task_yield(); }

fn main() { task::spawn((), yield_wrap); }
