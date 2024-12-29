//@ run-fail
//@ check-run-results:thread 'main' panicked
//@ check-run-results:foobar
//@ ignore-emscripten no processes

use std::panic;

fn main() {
    panic::take_hook();
    panic!("foobar");
}
