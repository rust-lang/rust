// compile-fail
// edition:2018
// compile-flags: --crate-type lib

#![feature(async_await)]

pub async const fn x() {}
//~^ ERROR expected one of `fn` or `unsafe`, found `const`
