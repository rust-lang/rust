//@ edition:2018
//@ compile-flags: --crate-type lib

pub async const fn x() {}
//~^ ERROR expected one of `extern`, `fn`, `safe`, or `unsafe`, found keyword `const`
//~| ERROR functions cannot be both `const` and `async`
