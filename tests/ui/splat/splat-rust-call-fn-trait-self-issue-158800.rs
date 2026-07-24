//! Regression test for `#[arg_splat] self` in a rust-call Fn* trait method not ICEing in WF
//! checking.

#![feature(arg_splat)]
#![feature(unboxed_closures)]
#![expect(incomplete_features)]

impl<T> dyn FnOnce(T) -> () {
    //~^ ERROR cannot define inherent `impl` for a type outside of the crate
    extern "rust-call" fn call_once(#[arg_splat] self) {}
    //~^ ERROR `#[arg_splat]` is not allowed in the arguments of functions with the `rust-call` ABI
    //~| ERROR functions with the "rust-call" ABI must take a single non-self tuple argument
    //~| ERROR the size for values of type `(dyn FnOnce(T) + 'static)` cannot be known
}

fn main() {}
