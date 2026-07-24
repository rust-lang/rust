//! Regression test for `#[arg_splat] ()` in a rust-call trait method not ICEing in WF
//! checking.

#![feature(arg_splat)]
#![feature(unboxed_closures)]
#![expect(incomplete_features)]

trait Trait {
    extern "rust-call" fn f(#[arg_splat] _: ()) where Self: Sized;
    //~^ ERROR `#[arg_splat]` is not allowed in the arguments of functions with the `rust-call` ABI
}

impl dyn Trait {
    extern "rust-call" fn f(#[arg_splat] _: ()) where Self: Sized {}
    //~^ ERROR `#[arg_splat]` is not allowed in the arguments of functions with the `rust-call` ABI
    //~| ERROR the size for values of type `(dyn Trait + 'static)` cannot be known at compilation time
}

fn main() {}
