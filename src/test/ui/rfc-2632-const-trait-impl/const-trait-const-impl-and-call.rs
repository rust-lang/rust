// check-pass
#![feature(const_trait_impl)]

#[const_trait]
trait Foo {
    fn foo(&self);
}

struct S;

impl const Foo for S {
    fn foo(&self) {}
}

const FOO: () = S.foo();

fn main() {}