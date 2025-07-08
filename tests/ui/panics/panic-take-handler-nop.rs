//@ run-fail
//@ check-run-results
//@ needs-subprocess

use std::panic;

fn main() {
    panic::take_hook();
    panic!("foobar");
}
