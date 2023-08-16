// run-fail
//@error-in-other-file:greetings from the panic handler
//@ignore-target-emscripten no processes

use std::panic;

fn main() {
    panic::set_hook(Box::new(|i| {
        eprintln!("greetings from the panic handler");
    }));
    panic!("foobar");
}
