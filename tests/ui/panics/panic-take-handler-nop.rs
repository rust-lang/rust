//@ run-fail
//@ error-pattern:thread 'main' panicked
//@ error-pattern:foobar
//@ needs-subprocess

use std::panic;

fn main() {
    panic::take_hook();
    panic!("foobar");
}
