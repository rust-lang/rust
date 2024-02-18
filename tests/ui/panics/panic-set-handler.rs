//@ run-fail
//@ error-pattern:greetings from the panic handler
//@ ignore-emscripten no processes

use std::panic;

fn main() {
    panic::set_hook(Box::new(|i| {
        eprintln!("greetings from the panic handler");
    }));
    panic!("foobar");
}
