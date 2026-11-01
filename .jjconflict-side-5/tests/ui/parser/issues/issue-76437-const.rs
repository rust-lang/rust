//@ edition:2018

mod t {
    const pub fn t() {}
    //~^ ERROR expected one of `async`, `extern`, `fn`, `safe`, or `unsafe`, found keyword `pub`
    //~| HELP visibility `pub` must come before `const`
}
