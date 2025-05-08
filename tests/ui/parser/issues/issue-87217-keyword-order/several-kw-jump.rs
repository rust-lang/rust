//@ edition:2018

// There is an order to respect for keywords before a function:
// `<visibility>, const, async, unsafe, extern, "<ABI>"`
//
// This test ensures the compiler is helpful about them being misplaced.
// Visibilities are tested elsewhere.

async unsafe const fn test() {}
//~^ ERROR expected one of `extern` or `fn`, found keyword `const`
//~| NOTE expected one of `extern` or `fn`
//~| HELP `const` must come before `async unsafe`
//~| SUGGESTION const async unsafe
//~| NOTE keyword order for functions declaration is `pub`, `default`, `const`, `async`, `unsafe`, `extern`
//~| ERROR functions cannot be both `const` and `async`
//~| NOTE `const` because of this
//~| NOTE `async` because of this

fn main() {}
