#![feature(rustc_attrs)]

trait A {
    fn foo_a(&self) {}
}

trait B {
    fn foo_b(&self) {}
}

trait C: A + B {
    fn foo_c(&self) {}
}

struct S;

#[rustc_dump_vtable]
impl A for S {}
//~^ error vtable

#[rustc_dump_vtable]
impl B for S {}
//~^ error vtable

#[rustc_dump_vtable]
impl C for S {}
//~^ error vtable

fn main() {}
