//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver

// Regression test for <https://github.com/rust-lang/rust/issues/158064>

fn foo() -> impl Sized {}

fn proj<T: FnOnce() -> U, U>(x: Option<T>, y: U) {}

fn main() {
    let mut x = None;
    proj(x, 1);
    //~^ ERROR: expected `foo` to return `{integer}`, but it returns `impl Sized`
    x = Some(foo);
}
