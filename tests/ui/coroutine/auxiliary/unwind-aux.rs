//@ compile-flags: -Cpanic=unwind  --crate-type=lib
//@ no-prefer-dynamic
//@ edition:2021

#![feature(coroutines)]
pub fn run<T>(a: T) {
    let _ = move || {
        drop(a);
        yield;
    };
}
