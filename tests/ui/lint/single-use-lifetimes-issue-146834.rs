//! Regression test for https://github.com/rust-lang/rust/issues/146834.

//@ compile-flags: -Wsingle-use-lifetimes
//@ edition: 2024

#![expect(incomplete_features)]
#![feature(contracts)]

#[core::contracts::ensures]
//~^ ERROR expected an `Fn(&_)` closure, found `()`
fn f<'a, 'b>(a: &'a i32, b: &'b i32) -> (&i32, &i32) {
    //~^ ERROR missing lifetime specifiers
    //~| WARN lifetime parameter `'a` only used once
    //~| WARN lifetime parameter `'b` only used once
    loop {}
}

fn main() {}
