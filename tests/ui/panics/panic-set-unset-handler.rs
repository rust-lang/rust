// run-fail
//@error-in-other-file:thread 'main' panicked at 'foobar'
//@ignore-target-emscripten no processes

use std::panic;

fn main() {
    panic::set_hook(Box::new(|i| {
        eprint!("greetings from the panic handler");
    }));
    panic::take_hook();
    panic!("foobar");
}
