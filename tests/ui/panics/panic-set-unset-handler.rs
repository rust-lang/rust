//@ run-fail
//@ check-run-results:thread 'main' panicked
//@ check-run-results:foobar
//@ ignore-emscripten no processes

use std::panic;

fn main() {
    panic::set_hook(Box::new(|i| {
        eprint!("greetings from the panic handler");
    }));
    panic::take_hook();
    panic!("foobar");
}
