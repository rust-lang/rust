//@ run-fail
//@ error-pattern:thread 'main' panicked
//@ error-pattern:foobar
//@ needs-subprocess

use std::panic;

fn main() {
    panic::set_hook(Box::new(|i| {
        eprint!("greetings from the panic handler");
    }));
    panic::take_hook();
    panic!("foobar");
}
