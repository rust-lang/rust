// edition:2018
// compile-flags: --crate-type lib

pub async const fn x() {}
//~^ ERROR expected one of `extern`, `fn`, or `unsafe`, found keyword `const`
