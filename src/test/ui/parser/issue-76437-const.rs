// edition:2018

mod t {
    const pub fn t() {}
    //~^ ERROR expected one of `async`, `extern`, `fn`, or `unsafe`, found keyword `pub`
    //~| HELP visibility `pub` must come before `const`
}
