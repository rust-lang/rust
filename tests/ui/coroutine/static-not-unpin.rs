//@ revisions: current next
//@[next] compile-flags: -Znext-solver

#![feature(coroutines)]

//@ normalize-stderr-test "std::pin::Unpin" -> "std::marker::Unpin"

use std::marker::Unpin;

fn assert_unpin<T: Unpin>(_: T) {
}

fn main() {
    let mut coroutine = static || {
        yield;
    };
    assert_unpin(coroutine); //~ ERROR E0277
}
