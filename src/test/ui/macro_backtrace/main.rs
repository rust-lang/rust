// Test that the macro backtrace facility works
// aux-build:ping.rs
// compile-flags: -Z external-macro-backtrace

#[macro_use] extern crate ping;

// a local macro
macro_rules! pong {
    () => { syntax error };
}
//~^^ ERROR expected one of
//~| ERROR expected one of
//~| ERROR expected one of

fn main() {
    pong!();
    ping!();
    deep!();
}
