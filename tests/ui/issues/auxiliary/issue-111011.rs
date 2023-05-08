#![feature(async_closure)]

// edition:2021

fn foo<X>(x: impl FnOnce() -> Box<X>) {}
// just to make sure async closures can still be suggested for boxing.
fn bar<X>(x: Box<dyn FnOnce() -> X>) {}

fn main() {
    foo(async move || {}); //~ ERROR mismatched types
    bar(async move || {}); //~ ERROR mismatched types
}
