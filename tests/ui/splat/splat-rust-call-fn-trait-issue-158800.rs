//! Regression test for `#[arg_splat] ()` in a rust-call Fn* trait method not ICEing in WF
//! checking.

#![feature(arg_splat)]
#![feature(unboxed_closures)]
#![expect(incomplete_features)]

impl<T> dyn FnOnce(T) -> () {
    //~^ ERROR cannot define inherent `impl` for a type outside of the crate
    extern "rust-call" fn call_once(#[arg_splat] _: ()) {}
    //~^ ERROR `#[arg_splat]` is not allowed in the arguments of functions with the `rust-call` ABI
}

fn main() {}
