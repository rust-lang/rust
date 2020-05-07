// Issue #7580

// run-fail
// error-pattern:panic works
// ignore-emscripten no processes

use std::*;

fn main() {
    panic!("panic works")
}
