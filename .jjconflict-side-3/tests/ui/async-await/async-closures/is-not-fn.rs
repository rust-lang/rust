//@ edition:2021
//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver

fn main() {
    fn needs_fn(x: impl FnOnce()) {}
    needs_fn(async || {});
    //~^ ERROR expected `{async closure@is-not-fn.rs:8:14}` to return `()`
}
