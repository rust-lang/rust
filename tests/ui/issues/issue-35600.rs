//@ run-pass
#![allow(non_camel_case_types)]
#![allow(unused_variables)]

trait Foo {
    type bar;
    fn bar();
}

impl Foo for () {
    type bar = ();
    fn bar() {}
}

fn main() {
    let x: <() as Foo>::bar = ();
    <()>::bar();
}
