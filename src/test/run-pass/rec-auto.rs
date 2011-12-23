


// -*- rust -*-

// Issue #50.
fn main() {
    let x = {foo: "hello", bar: "world"};
    log(debug, x.foo);
    log(debug, x.bar);
}
