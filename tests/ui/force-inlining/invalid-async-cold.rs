//@ edition: 2024
//@ compile-flags: --crate-type=lib
#![allow(internal_features)]
#![feature(gen_blocks)]
#![feature(rustc_attrs)]

// `#[inline]` on an `async fn` or `gen fn` applies to its body coroutine, but
// `#[rustc_force_inline]` must not: it is already rejected on such functions,
// so it must not also be reported as incompatible with `#[cold]`, which both
// the function and its body coroutine get.

#[rustc_force_inline]
//~^ ERROR attribute cannot be applied to a `async`, `gen` or `async gen` function
#[cold]
pub async fn async_cold() {}

#[rustc_force_inline]
//~^ ERROR attribute cannot be applied to a `async`, `gen` or `async gen` function
#[cold]
pub gen fn gen_cold() {}

pub fn caller() {
    let _ = async_cold();
    let _ = gen_cold();
}
