// Test for issue https://github.com/rust-lang/rust/issues/140171
// There is an order to respect for keywords before a function:
// `<visibility>, const, async, unsafe, extern, "<ABI>"`
//
// This test ensures the compiler is helpful about them being misplaced.
//@ edition:2018

extern "C" const async unsafe fn b() {}
//~^ ERROR expected `fn`, found keyword `const`
//~| NOTE expected `fn`
//~| HELP `const` must come before `extern "C"`
//~| SUGGESTION const extern "C"
//~| NOTE keyword order for functions declaration is `pub`, `default`, `const`, `async`, `unsafe`, `extern`

fn main() {}
