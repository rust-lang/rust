/*
  A reduced test case for Issue #506, provided by Rob Arnold.
*/

use std;
import std::task;

native "rust" mod rustrt {
    fn task_yield();
}

fn yield_wrap() { rustrt::task_yield(); }

fn main() { let f = yield_wrap; task::spawn(f); }
