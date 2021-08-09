pub const pub fn test() {}
//~^ ERROR expected one of `async`, `extern`, `fn`, or `unsafe`, found keyword `pub`
//~| NOTE expected one of `async`, `extern`, `fn`, or `unsafe`
//~| HELP there is already a visibility, remove this one
//~| NOTE explicit visibility first seen here
