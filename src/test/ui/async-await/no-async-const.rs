// edition:2018
// compile-flags: --crate-type lib

pub async const fn x() {}
//~^ ERROR expected one of `fn` or `unsafe`, found keyword `const`
