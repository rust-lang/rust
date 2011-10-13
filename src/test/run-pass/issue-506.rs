/*
  A reduced test case for Issue #506, provided by Rob Arnold.
*/

use std;
import std::task;

native "rust" mod rustrt {
    fn task_yield();
}

fn# yield_wrap(&&_i: ()) unsafe { rustrt::task_yield(); }

fn main() { task::spawn2((), yield_wrap); }
