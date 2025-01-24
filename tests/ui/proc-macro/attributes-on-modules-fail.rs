//@ proc-macro: test-macros.rs

#[macro_use]
extern crate test_macros;

#[identity_attr]
mod m {
    pub struct X;

    type A = Y; //~ ERROR cannot find type `Y` in this scope
}

struct Y;
type A = X; //~ ERROR cannot find type `X` in this scope

#[derive(Copy)] //~ ERROR `derive` may only be applied to `struct`s, `enum`s and `union`s
mod n {}

#[empty_attr]
mod module; //~ ERROR non-inline modules in proc macro input are unstable

#[empty_attr]
mod outer {
    mod inner; //~ ERROR non-inline modules in proc macro input are unstable

    mod inner_inline {} // OK
}

#[derive(Empty)]
struct S {
    field: [u8; {
        #[path = "outer/inner.rs"]
        mod inner; //~ ERROR non-inline modules in proc macro input are unstable
        mod inner_inline {} // OK
        0
    }]
}

#[identity_attr]
fn f() {
    #[path = "outer/inner.rs"]
    mod inner; //~ ERROR non-inline modules in proc macro input are unstable
    mod inner_inline {} // OK
}

fn main() {}
