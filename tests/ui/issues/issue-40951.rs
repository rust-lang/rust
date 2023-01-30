// run-pass
#![allow(unused_variables)]
// Regression test for #40951.

const FOO: [&'static str; 1] = ["foo"];

fn find<T: PartialEq>(t: &[T], element: &T) { }

fn main() {
    let x = format!("hi");
    find(&FOO, &&*x);
}
