// edition:2021

#![feature(async_closure)]

fn main() {
    fn needs_fn(x: impl FnOnce()) {}
    needs_fn(async || {});
    //~^ ERROR expected `{coroutine-closure@is-not-fn.rs:7:14}` to be a closure that returns `()`
}
