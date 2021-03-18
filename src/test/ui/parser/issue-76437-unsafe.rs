// edition:2018

mod t {
    unsafe pub fn t() {}
    //~^ ERROR expected one of `extern` or `fn`, found keyword `pub`
    //~| HELP visibility `pub` must come before `unsafe`
}
