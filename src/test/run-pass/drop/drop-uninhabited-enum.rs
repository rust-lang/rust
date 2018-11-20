// run-pass
#![allow(dead_code)]
#![allow(unused_variables)]
#![allow(unreachable_code)]

enum Foo {}

impl Drop for Foo {
    fn drop(&mut self) { }
}

fn foo() {
    let _x: Foo = unimplemented!();
}

fn main() { }
