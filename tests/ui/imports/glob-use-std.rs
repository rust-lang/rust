// Issue #7580

// run-fail
//@error-in-other-file:panic works
//@ignore-target-emscripten no processes

use std::*;

fn main() {
    panic!("panic works")
}
