//@ edition:2018

mod t {
    const async unsafe pub fn t() {}
    //~^ ERROR expected one of `extern` or `fn`, found keyword `pub`
    //~| HELP visibility `pub` must come before `const async unsafe`
}
