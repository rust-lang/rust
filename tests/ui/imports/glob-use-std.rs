// Issue #7580

//@ run-fail
//@ check-run-results:panic works
//@ ignore-emscripten no processes

use std::*;

fn main() {
    panic!("panic works")
}
