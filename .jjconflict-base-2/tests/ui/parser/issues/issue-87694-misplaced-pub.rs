const pub fn test() {}
//~^ ERROR expected one of `async`, `extern`, `fn`, `safe`, or `unsafe`, found keyword `pub`
//~| NOTE expected one of `async`, `extern`, `fn`, `safe`, or `unsafe`
//~| HELP visibility `pub` must come before `const`
//~| SUGGESTION pub const
