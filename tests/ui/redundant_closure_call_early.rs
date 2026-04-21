// non rustfixable, see redundant_closure_call_fixable.rs

#![expect(incomplete_features)]
#![feature(ergonomic_clones)]
#![warn(clippy::redundant_closure_call)]

fn main() {
    let mut i = 1;

    // lint here
    let mut k = (|m| m + 1)(i);
    //~^ redundant_closure_call

    // lint here
    k = (|a, b| a * b)(1, 5);
    //~^ redundant_closure_call

    // don't lint these
    #[allow(clippy::needless_return)]
    (|| return 2)();
    (|| -> Option<i32> { None? })();
    #[allow(clippy::try_err)]
    (|| -> Result<i32, i32> { Err(2)? })();

    // don't lint async equivalents either
    #[expect(clippy::needless_return)]
    (|| async { return })();
    (|| async {
        let x: Option<i32> = None;
        x?;
        Some(1)
    })();
    #[expect(clippy::try_err)]
    (|| async {
        Err::<(), i32>(2)?;
        Ok::<(), i32>(())
    })();

    #[expect(clippy::needless_return)]
    (|| async move { return })();
    (|| async move {
        let x: Option<i32> = None;
        x?;
        Some(1)
    })();

    #[expect(clippy::needless_return)]
    (async || return)();
    (async || {
        let x: Option<i32> = None;
        x?;
        Some(1)
    })();
    #[expect(clippy::try_err)]
    (async || {
        Err::<(), i32>(2)?;
        Ok::<(), i32>(())
    })();

    #[expect(clippy::needless_return)]
    (async move || return)();
    (async move || {
        let x: Option<i32> = None;
        x?;
        Some(1)
    })();

    #[expect(clippy::needless_return)]
    (async use || return)();
    (async use || {
        let x: Option<i32> = None;
        x?;
        Some(1)
    })();
}
