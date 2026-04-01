//@ edition:2018

// Test that even when `const` is already present, the proposed fix is to remove the second `const`

const async const fn test() {}
//~^ ERROR expected one of `extern`, `fn`, `safe`, or `unsafe`, found keyword `const`
//~| NOTE expected one of `extern`, `fn`, `safe`, or `unsafe`
//~| HELP `const` already used earlier, remove this one
//~| NOTE `const` first seen here
//~| ERROR functions cannot be both `const` and `async`
//~| NOTE `const` because of this
//~| NOTE `async` because of this

fn main() {}
