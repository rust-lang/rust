//@ run-fail
//@ check-run-results
//@ needs-subprocess

use std::panic;

fn main() {
    panic::set_hook(Box::new(|i| {
        eprint!("greetings from the panic handler");
    }));
    panic::take_hook();
    panic!("foobar");
}
