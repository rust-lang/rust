// check-pass
#![feature(const_trait_impl)]

trait Foo {
    fn a(&self);
}
trait Bar: ~const Foo {}

struct S;
impl const Foo for S {
    fn a(&self) {}
}

impl const Bar for S {}

const fn foo<T: ~const Bar>(t: &T) {
    t.a();
}

const _: () = foo(&S);

fn main() {}
