//@ run-fail
//@ regex-error-pattern: thread 'main' \(\d+\) panicked
//@ error-pattern: foobar
//@ needs-subprocess

use std::panic;

fn main() {
    panic::take_hook();
    panic!("foobar");
}
