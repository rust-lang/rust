//@ edition:2018

mod t {
    async pub fn t() {}
    //~^ ERROR expected one of `extern`, `fn`, `safe`, or `unsafe`, found keyword `pub`
    //~| HELP visibility `pub` must come before `async`
}
