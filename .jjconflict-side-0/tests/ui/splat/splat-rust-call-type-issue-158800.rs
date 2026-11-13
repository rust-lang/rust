//! Regression test for `#[rustc_splat] ())` in a rust-call type method not ICEing in WF
//! checking.

#![feature(splat)]
#![feature(unboxed_closures)]
#![expect(incomplete_features)]

struct Type;

trait Trait {
    extern "rust-call" fn f(#[rustc_splat] _: ());
    //~^ ERROR `#[rustc_splat]` is not allowed in the arguments of functions with the `rust-call` ABI
}

impl Type {
    extern "rust-call" fn f2(#[rustc_splat] _: ()) {}
    //~^ ERROR `#[rustc_splat]` is not allowed in the arguments of functions with the `rust-call` ABI
}

impl Trait for Type {
    extern "rust-call" fn f(#[rustc_splat] _: ()) {}
    //~^ ERROR `#[rustc_splat]` is not allowed in the arguments of functions with the `rust-call` ABI
}

fn main() {}
