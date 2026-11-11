//@ edition:2018

// There is an order to respect for keywords before a function:
// `<visibility>, const, async, unsafe, extern, "<ABI>"`
//
// This test ensures the compiler is helpful about them being misplaced.
// Visibilities are tested elsewhere.

unsafe async fn test() {}
//~^ ERROR expected one of `extern` or `fn`, found keyword `async`
//~| NOTE expected one of `extern` or `fn`
//~| HELP `async` must come before `unsafe`
//~| SUGGESTION async unsafe
//~| NOTE keyword order for functions declaration is `pub`, `default`, `const`, `async`, `unsafe`, `extern`

fn main() {}
