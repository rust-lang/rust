// edition:2018

// Test that even when `const` is already present, the proposed fix is to remove the second `const`

const async const fn test() {}
//~^ ERROR expected one of `extern`, `fn`, or `unsafe`, found keyword `const`
//~| NOTE expected one of `extern`, `fn`, or `unsafe`
//~| HELP `const` already used earlier, remove this one
//~| NOTE `const` first seen here
