#![allow(bare_trait_objects)]

trait DynIncompatible {
    fn foo() -> Self;
}

struct A;
struct B;

impl DynIncompatible for A {
    fn foo() -> Self {
        A
    }
}

impl DynIncompatible for B {
    fn foo() -> Self {
        B
    }
}

fn car() -> dyn DynIncompatible { //~ ERROR the trait `DynIncompatible` is not dyn compatible
//~^ ERROR return type cannot be a trait object without pointer indirection
    if true {
        return A;
    }
    B
}

fn cat() -> Box<dyn DynIncompatible> { //~ ERROR the trait `DynIncompatible` is not dyn compatible
    if true {
        return Box::new(A);
    }
    Box::new(B)
}

fn main() {}
