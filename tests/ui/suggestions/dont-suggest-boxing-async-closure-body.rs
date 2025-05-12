//@ edition:2021
// issue: https://github.com/rust-lang/rust/issues/111011

fn foo<X>(x: impl FnOnce() -> Box<X>) {}
// just to make sure async closures can still be suggested for boxing.
fn bar<X>(x: Box<dyn FnOnce() -> X>) {}

fn main() {
    foo(async move || {});
    //~^ ERROR expected `{async closure@dont-suggest-boxing-async-closure-body.rs:9:9}` to return `Box<_>`
    bar(async move || {}); //~ ERROR mismatched types
}
