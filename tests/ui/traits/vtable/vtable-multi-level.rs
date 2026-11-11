#![feature(rustc_attrs)]

//   O --> G --> C --> A
//     \     \     \-> B
//     |     |-> F --> D
//     |           \-> E
//     |-> N --> J --> H
//           \     \-> I
//           |-> M --> K
//                 \-> L

trait A {
    fn foo_a(&self) {}
}

trait B {
    fn foo_b(&self) {}
}

trait C: A + B {
    fn foo_c(&self) {}
}

trait D {
    fn foo_d(&self) {}
}

trait E {
    fn foo_e(&self) {}
}

trait F: D + E {
    fn foo_f(&self) {}
}

trait G: C + F {
    fn foo_g(&self) {}
}

trait H {
    fn foo_h(&self) {}
}

trait I {
    fn foo_i(&self) {}
}

trait J: H + I {
    fn foo_j(&self) {}
}

trait K {
    fn foo_k(&self) {}
}

trait L {
    fn foo_l(&self) {}
}

trait M: K + L {
    fn foo_m(&self) {}
}

trait N: J + M {
    fn foo_n(&self) {}
}

trait O: G + N {
    fn foo_o(&self) {}
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

#[rustc_dump_vtable]
impl E for S {}
//~^ ERROR vtable entries

#[rustc_dump_vtable]
impl F for S {}
//~^ ERROR vtable entries

#[rustc_dump_vtable]
impl G for S {}
//~^ ERROR vtable entries

#[rustc_dump_vtable]
impl H for S {}
//~^ ERROR vtable entries

#[rustc_dump_vtable]
impl I for S {}
//~^ ERROR vtable entries

#[rustc_dump_vtable]
impl J for S {}
//~^ ERROR vtable entries

#[rustc_dump_vtable]
impl K for S {}
//~^ ERROR vtable entries

#[rustc_dump_vtable]
impl L for S {}
//~^ ERROR vtable entries

#[rustc_dump_vtable]
impl M for S {}
//~^ ERROR vtable entries

#[rustc_dump_vtable]
impl N for S {}
//~^ ERROR vtable entries

#[rustc_dump_vtable]
impl O for S {}
//~^ ERROR vtable entries

fn main() {}
