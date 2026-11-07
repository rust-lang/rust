//@ edition:2018

// There is an order to respect for keywords before a function:
// `<visibility>, const, async, unsafe, extern, "<ABI>"`
//
// This test ensures the compiler is helpful about them being misplaced.
// Visibilities are tested elsewhere.

extern "C" unsafe fn test() {}
//~^ ERROR expected `fn`, found keyword `unsafe`
//~| NOTE expected `fn`
//~| HELP `unsafe` must come before `extern "C"`
//~| SUGGESTION unsafe extern "C"
//~| NOTE keyword order for functions declaration is `pub`, `default`, `const`, `async`, `unsafe`, `extern`

fn main() {}
