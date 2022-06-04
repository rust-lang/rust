// check-pass
#![feature(const_trait_impl)]

#[const_trait]
trait Foo {
    fn foo(&self);
}

pub struct S;

impl Foo for S {
    fn foo(&self) {}
}

fn non_const() {
    S.foo();
}

fn main() {}