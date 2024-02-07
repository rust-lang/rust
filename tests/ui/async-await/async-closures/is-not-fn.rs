// edition:2021

#![feature(async_closure)]

fn main() {
    fn needs_fn(x: impl FnOnce()) {}
    needs_fn(async || {});
    //~^ ERROR expected a `FnOnce()` closure, found `{coroutine-closure@
    // FIXME(async_closures): This should explain in more detail how async fns don't
    // implement the regular `Fn` traits. Or maybe we should just fix it and make them
    // when there are no upvars or whatever.
}
