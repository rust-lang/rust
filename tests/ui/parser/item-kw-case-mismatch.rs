//@ run-rustfix
//@ edition:2018
#![allow(unused_imports)]

fn main() {}

Use std::ptr::read;  //~ ERROR keyword `use` is written in the wrong case
USE std::ptr::write; //~ ERROR keyword `use` is written in the wrong case

async Fn _a() {}
//~^ ERROR keyword `fn` is written in the wrong case

Fn _b() {}
//~^ ERROR keyword `fn` is written in the wrong case

aSYNC fN _c() {}
//~^ ERROR keyword `async` is written in the wrong case
//~| ERROR keyword `fn` is written in the wrong case

Async fn _d() {}
//~^ ERROR keyword `async` is written in the wrong case

CONST UNSAFE FN _e() {}
//~^ ERROR keyword `const` is written in the wrong case
//~| ERROR keyword `unsafe` is written in the wrong case
//~| ERROR keyword `fn` is written in the wrong case

unSAFE EXTern "C" fn _f() {}
//~^ ERROR keyword `unsafe` is written in the wrong case
//~| ERROR keyword `extern` is written in the wrong case

EXTERN "C" FN _g() {}
//~^ ERROR keyword `extern` is written in the wrong case
//~| ERROR keyword `fn` is written in the wrong case
