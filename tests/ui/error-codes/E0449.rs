//@ run-rustfix

#![allow(warnings)]

struct Bar;

trait Foo {
    fn foo();
}

pub impl Bar {} //~ ERROR E0449

pub impl Foo for Bar { //~ ERROR E0449
    pub fn foo() {} //~ ERROR E0449
}

fn main() {
}
