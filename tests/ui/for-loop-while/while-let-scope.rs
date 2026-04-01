//@ check-pass
// regression test for #40235
#![allow(unused_variables)]
fn foo() {}

fn main() {
    while let Some(foo) = Some(1) { break }
    foo();
}
