// edition:2018

// There is an order to respect for keywords before a function:
// `<visibility>, const, async, unsafe, extern, "<ABI>"`
//
// This test ensures the compiler is helpful about them being misplaced.
// Visibilities are tested elsewhere.

trait MyT {
    unsafe fn test();
}

impl<T> MyT for T {
    unsafe default fn test() {}
    //~^ ERROR expected one of `extern` or `fn`, found keyword `default`
    //~| NOTE expected one of `extern` or `fn`
    //~| HELP `default` must come before `unsafe`
    //~| SUGGESTION default unsafe
    //~| NOTE keyword order for functions declaration is `default|pub`, `const`, `async`, `unsafe`, `extern`
}

fn main() {}
