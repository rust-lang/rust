


// -*- rust -*-

// Issue #50.
fn main() {
    let x = {foo: ~"hello", bar: ~"world"};
    log(debug, copy x.foo);
    log(debug, copy x.bar);
}
