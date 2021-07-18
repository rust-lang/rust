// edition:2018

// There is an order to respect for keywords before a function:
// `<visibility>, const, async, unsafe, extern, "<ABI>"`
//
// This test ensures the compiler is helpful about them being misplaced.
// Visibilities are tested elsewhere.

extern "C" const fn test() {}
//~^ ERROR expected `fn`, found keyword `const`
//~| NOTE expected `fn`
//~| HELP `const` must come before `extern "C"`
//~| SUGGESTION const extern "C"
//~| NOTE keyword order for functions declaration is `default|pub`, `const`, `async`, `unsafe`, `extern`
