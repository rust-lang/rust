//@ add-core-stubs
//@ compile-flags: --target thumbv8m.main-none-eabi --crate-type lib
//@ incremental (required to trigger the bug)
//@ needs-llvm-components: arm
#![feature(abi_cmse_nonsecure_call, no_core)]
#![no_core]

extern crate minicore;
use minicore::*;

// A regression test for https://github.com/rust-lang/rust/issues/131639.
// NOTE: `-Cincremental` was required for triggering the bug.

fn foo() {
    id::<extern "cmse-nonsecure-call" fn(&'a ())>(PhantomData);
    //~^ ERROR use of undeclared lifetime name `'a`
}

fn id<T>(x: PhantomData<T>) -> PhantomData<T> {
    x
}
