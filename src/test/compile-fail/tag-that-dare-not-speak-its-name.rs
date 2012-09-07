// error-pattern:mismatched types: expected `char` but found
// Issue #876

#[no_core];

use core;

fn last<T: Copy>(v: ~[const T]) -> core::Option<T> {
    fail;
}

fn main() {
    let y;
    let x : char = last(y);
}
