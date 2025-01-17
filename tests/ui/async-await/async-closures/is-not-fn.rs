//@ edition:2021

fn main() {
    fn needs_fn(x: impl FnOnce()) {}
    needs_fn(async || {});
    //~^ ERROR expected `{async closure@is-not-fn.rs:5:14}` to be a closure that returns `()`
}
