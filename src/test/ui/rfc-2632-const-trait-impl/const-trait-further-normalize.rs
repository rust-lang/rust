// check-pass
#![feature(const_trait_impl)]

#[const_trait]
trait Foo {
    fn foo(&self);
}

struct S1;
struct S2;

impl Foo for S1 {
    fn foo(&self) {}
}

impl const Foo for S2 where S1: ~const Foo {
    fn foo(&self) {}
}

fn main() {
    S2.foo();
}

