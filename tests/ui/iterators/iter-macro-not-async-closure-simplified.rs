// This test ensures iterators created with the `iter!` macro are not
// accidentally async closures.
//
// We test this both in a `narrow` and `wide` configuration because
// the way that the diagnostic is emitted varies depending on the
// diagnostic width.  If it's too narrow to fit the explanation, that
// explanation is moved to the `help` instead of the span label.
//
//@ edition: 2024
//@ revisions: narrow wide
//@[narrow] compile-flags: --diagnostic-width=20
//@[wide] compile-flags: --diagnostic-width=300

#![feature(yield_expr, iter_macro)]

use std::iter::iter;

fn call_async_once(_: impl AsyncFnOnce()) {}

fn main() {
    let f = iter! { move || {
        for i in 0..10 {
            yield i;
        }
    }};

    call_async_once(f);
    //~^ ERROR AsyncFnOnce()` is not satisfied
}
