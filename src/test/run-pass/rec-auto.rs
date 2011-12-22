


// -*- rust -*-

// Issue #50.
fn main() {
    let x = {foo: "hello", bar: "world"};
    log_full(core::debug, x.foo);
    log_full(core::debug, x.bar);
}
