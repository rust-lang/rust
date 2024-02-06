// edition:2021

// FIXME(async_closures): This needs a better error message!

#![feature(async_closure)]

fn main() {
    fn needs_async_fn(_: impl async Fn()) {}

    let mut x = 1;
    needs_async_fn(async || {
        //~^ ERROR i16: ops::async_function::internal_implementation_detail::AsyncFnKindHelper<i8>
        // FIXME: Should say "closure is `async FnMut` but it needs `async Fn`" or sth.
        x += 1;
    });
}
