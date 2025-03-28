// This test ensures iterators created with the `iter!` macro are not accidentally async closures.
//@ edition: 2024

#![feature(yield_expr, iter_macro)]

use std::iter::iter;

fn call_async_once(_: impl AsyncFnOnce()) {
}

fn main() {
    let f = iter! { move || {
        for i in 0..10 {
            yield i;
        }
    }};

    call_async_once(f);
    //~^ ERROR AsyncFnOnce()` is not satisfied
}
