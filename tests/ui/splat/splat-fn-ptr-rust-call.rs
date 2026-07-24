//! Test using `#[arg_splat]` on tuple arguments of pointers to "rust-call" functions.
//! Currently ICEs at a later stage, but AST validation should catch it earlier.

#![allow(incomplete_features)]
#![feature(arg_splat)]
#![feature(unboxed_closures)]

extern "rust-call" fn f(#[arg_splat] _: ()) {} //~ ERROR `#[arg_splat]` is not allowed in the arguments of functions with the `rust-call` ABI

fn main() {
    // FIXME(rustfmt): the attribute gets deleted by rustfmt
    #[rustfmt::skip]
    let f2: extern "rust-call" fn(#[arg_splat] ()) = f; //~ ERROR `#[arg_splat]` is not allowed in the arguments of functions with the `rust-call` ABI
    // These errors could be confusing, but they're useful if the user meant to use "rust-call"
    // instead of #[arg_splat]
    f(); //~ ERROR functions with the "rust-call" ABI must take a single non-self tuple argument
    f2(); //~ ERROR functions with the "rust-call" ABI must take a single non-self tuple argument
}
