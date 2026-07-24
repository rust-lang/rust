//! Regression test for `#[arg_splat] ())` in a rust-call type method not ICEing in WF
//! checking.

#![feature(arg_splat)]
#![feature(unboxed_closures)]
#![expect(incomplete_features)]

struct Type;

trait Trait {
    extern "rust-call" fn f(#[arg_splat] _: ());
    //~^ ERROR `#[arg_splat]` is not allowed in the arguments of functions with the `rust-call` ABI
}

impl Type {
    extern "rust-call" fn f2(#[arg_splat] _: ()) {}
    //~^ ERROR `#[arg_splat]` is not allowed in the arguments of functions with the `rust-call` ABI
}

impl Trait for Type {
    extern "rust-call" fn f(#[arg_splat] _: ()) {}
    //~^ ERROR `#[arg_splat]` is not allowed in the arguments of functions with the `rust-call` ABI
}

fn main() {}
