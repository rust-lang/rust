// build-fail
#![feature(rustc_attrs)]

#[rustc_dump_vtable]
trait A {
    fn foo_a(&self) {}
}

#[rustc_dump_vtable]
trait B {
    //~^ error vtable
    fn foo_b(&self) {}
}

#[rustc_dump_vtable]
trait C: A + B {
    //~^ error vtable
    fn foo_c(&self) {}
}

struct S;

impl A for S {}
impl B for S {}
impl C for S {}

fn foo(c: &dyn C) {}

fn main() {
    foo(&S);
}
