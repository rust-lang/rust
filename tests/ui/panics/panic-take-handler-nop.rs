// run-fail
//@error-in-other-file:thread 'main' panicked at 'foobar'
//@ignore-target-emscripten no processes

use std::panic;

fn main() {
    panic::take_hook();
    panic!("foobar");
}
