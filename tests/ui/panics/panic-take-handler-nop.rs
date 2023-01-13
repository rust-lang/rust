// run-fail
// error-pattern:thread 'main' panicked at 'foobar'
// ignore-emscripten no processes

use std::panic;

fn main() {
    panic::take_hook();
    panic!("foobar");
}
