//@ build-fail
#![feature(rustc_attrs)]
#![feature(negative_impls)]

// B --> A

#[rustc_dump_vtable]
trait A {
    fn foo_a1(&self) {}
    fn foo_a2(&self) where Self: Send {}
}

#[rustc_dump_vtable]
trait B: A {
    //~^ error vtable
    fn foo_b1(&self) {}
    fn foo_b2(&self) where Self: Send {}
}

struct S;
impl !Send for S {}

impl A for S {}
impl B for S {}

fn foo(_: &dyn B) {}

fn main() {
    foo(&S);
}
