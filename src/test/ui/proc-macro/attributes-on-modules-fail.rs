// aux-build:test-macros.rs

#[macro_use]
extern crate test_macros;

#[identity_attr] //~ ERROR custom attributes cannot be applied to modules
mod m {
    pub struct X;

    type A = Y; //~ ERROR cannot find type `Y` in this scope
}

struct Y;
type A = X; //~ ERROR cannot find type `X` in this scope

#[derive(Copy)] //~ ERROR `derive` may only be applied to structs, enums and unions
mod n {}

#[empty_attr]
mod module; //~ ERROR non-inline modules in proc macro input are unstable

#[empty_attr] //~ ERROR custom attributes cannot be applied to modules
mod outer {
    mod inner; //~ ERROR non-inline modules in proc macro input are unstable

    mod inner_inline {} // OK
}

fn main() {}
