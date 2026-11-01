//@ edition:2021
//@ run-rustfix

fn foo<T>(_: Box<T>) {}
fn bar<T>(_: impl Fn() -> Box<T>) {}

fn main() {
    foo({}); //~ ERROR mismatched types
    bar(|| {}); //~ ERROR mismatched types
}
