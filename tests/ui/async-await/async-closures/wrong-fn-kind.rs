// edition:2021

// FIXME(async_closures): This needs a better error message!

#![feature(async_closure, async_fn_traits)]

use std::ops::AsyncFn;

fn main() {
    fn needs_async_fn(_: impl AsyncFn()) {}

    let mut x = 1;
    needs_async_fn(async || {
        //~^ ERROR i16: ops::async_function::internal_implementation_detail::AsyncFnKindHelper<i8>
        // FIXME: Should say "closure is AsyncFnMut but it needs AsyncFn" or sth.
        x += 1;
    });
}
