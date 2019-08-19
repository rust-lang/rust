// edition:2018

struct S;

impl S {
    #[cfg(FALSE)]
    unsafe async fn g() {} //~ ERROR expected one of `extern` or `fn`, found `async`
}

#[cfg(FALSE)]
unsafe async fn f() {} //~ ERROR expected one of `extern`, `fn`, or `{`, found `async`
