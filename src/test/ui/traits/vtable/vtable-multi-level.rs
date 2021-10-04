// build-fail
#![feature(rustc_attrs)]

//   O --> G --> C --> A
//     \     \     \-> B
//     |     |-> F --> D
//     |           \-> E
//     |-> N --> J --> H
//           \     \-> I
//           |-> M --> K
//                 \-> L

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
    fn foo_c(&self) {}
}

#[rustc_dump_vtable]
trait D {
    //~^ error vtable
    fn foo_d(&self) {}
}

#[rustc_dump_vtable]
trait E {
    //~^ error vtable
    fn foo_e(&self) {}
}

#[rustc_dump_vtable]
trait F: D + E {
    //~^ error vtable
    fn foo_f(&self) {}
}

#[rustc_dump_vtable]
trait G: C + F {
    fn foo_g(&self) {}
}

#[rustc_dump_vtable]
trait H {
    //~^ error vtable
    fn foo_h(&self) {}
}

#[rustc_dump_vtable]
trait I {
    //~^ error vtable
    fn foo_i(&self) {}
}

#[rustc_dump_vtable]
trait J: H + I {
    //~^ error vtable
    fn foo_j(&self) {}
}

#[rustc_dump_vtable]
trait K {
    //~^ error vtable
    fn foo_k(&self) {}
}

#[rustc_dump_vtable]
trait L {
    //~^ error vtable
    fn foo_l(&self) {}
}

#[rustc_dump_vtable]
trait M: K + L {
    //~^ error vtable
    fn foo_m(&self) {}
}

#[rustc_dump_vtable]
trait N: J + M {
    //~^ error vtable
    fn foo_n(&self) {}
}

#[rustc_dump_vtable]
trait O: G + N {
    //~^ error vtable
    fn foo_o(&self) {}
}

struct S;

impl A for S {}
impl B for S {}
impl C for S {}
impl D for S {}
impl E for S {}
impl F for S {}
impl G for S {}
impl H for S {}
impl I for S {}
impl J for S {}
impl K for S {}
impl L for S {}
impl M for S {}
impl N for S {}
impl O for S {}

fn foo(_: &dyn O) {}

fn main() {
    foo(&S);
}
