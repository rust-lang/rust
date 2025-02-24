// Issue #7580

//@ run-fail
//@ error-pattern:panic works
//@ needs-subprocess

use std::*;

fn main() {
    panic!("panic works")
}
