// run-pass
#![allow(unused_variables)]
fn foo() {}

fn main() {
    while let Some(foo) = Some(1) { break }
    foo();
}
