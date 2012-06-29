// error-pattern:mismatched types: expected `char` but found
// Issue #876

#[no_core];

use core;

fn last<T: copy>(v: ~[const T]) -> core::option<T> {
    fail;
}

fn main() {
    let y;
    let x : char = last(y);
}
