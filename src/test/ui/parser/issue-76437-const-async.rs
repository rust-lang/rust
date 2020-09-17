// edition:2018

mod t {
    const async pub fn t() {}
    //~^ ERROR expected one of `extern`, `fn`, or `unsafe`, found keyword `pub`
    //~| HELP visibility `pub` must come before `const async`
}
