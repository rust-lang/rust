#![feature(rustc_attrs)]

trait A {
    fn foo_a(&self) {}
}

trait B: A {
    fn foo_b(&self) {}
}

trait C: A {
    fn foo_c(&self) {}
}

trait D: B + C {
    fn foo_d(&self) {}
}

struct S;

#[rustc_dump_vtable]
impl A for S {}
//~^ ERROR vtable entries

#[rustc_dump_vtable]
impl B for S {}
//~^ ERROR vtable entries

#[rustc_dump_vtable]
impl C for S {}
//~^ ERROR vtable entries

#[rustc_dump_vtable]
impl D for S {}
//~^ ERROR vtable entries

fn main() {}
