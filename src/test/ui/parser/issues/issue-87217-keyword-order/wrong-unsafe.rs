// edition:2018

// There is an order to respect for keywords before a function:
// `<visibility>, const, async, unsafe, extern, "<ABI>"`
//
// This test ensures the compiler is helpful about them being misplaced.
// Visibilities are tested elsewhere.

extern unsafe fn test() {}
//~^ ERROR expected `fn`, found keyword `unsafe`
//~| NOTE expected `fn`
//~| HELP `unsafe` must come before `extern`
//~| SUGGESTION unsafe extern
//~| NOTE keyword order for functions declaration is `default`, `pub`, `const`, `async`, `unsafe`, `extern`
