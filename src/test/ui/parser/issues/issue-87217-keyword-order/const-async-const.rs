// edition:2018

// Test that even when `const` is already present, the proposed fix is `const const async`,
// like for `pub pub`.

const async const fn test() {}
//~^ ERROR expected one of `extern`, `fn`, or `unsafe`, found keyword `const`
//~| NOTE expected one of `extern`, `fn`, or `unsafe`
//~| HELP `const` must come before `async`
//~| SUGGESTION const async
//~| NOTE keyword order for functions declaration is `default`, `pub`, `const`, `async`, `unsafe`, `extern`
