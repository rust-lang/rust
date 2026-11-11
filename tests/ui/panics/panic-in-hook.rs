// Checks what happens when panicking inside the panic hook.

//@ run-crash
//@ exec-env:RUST_BACKTRACE=0
//@ check-run-results
//@ error-pattern: panicked while processing panic
//@ ignore-emscripten "RuntimeError" junk in output

use std::panic;

fn main() {
    panic::set_hook(Box::new(|_| panic!("panic in hook")));
    panic!();
}
