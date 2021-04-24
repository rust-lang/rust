// edition:2018

mod t {
    unsafe pub(crate) fn t() {}
    //~^ ERROR expected one of `extern` or `fn`, found keyword `pub`
    //~| HELP visibility `pub(crate)` must come before `unsafe`
}
