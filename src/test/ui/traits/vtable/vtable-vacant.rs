// build-fail
#![feature(rustc_attrs)]

// B --> A

#[rustc_dump_vtable]
trait A {
    fn foo_a1(&self) {}
    fn foo_a2(&self) where Self: Sized {}
}

#[rustc_dump_vtable]
trait B: A {
    //~^ error Vtable
    fn foo_b1(&self) {}
    fn foo_b2() where Self: Sized {}
}

struct S;

impl A for S {}
impl B for S {}

fn foo(_: &dyn B) {}

fn main() {
    foo(&S);
}
