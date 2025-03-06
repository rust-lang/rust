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
//~^ ERROR return type cannot have an unboxed trait object
    if true {
        return A;
    }
    B
}

fn cat() -> Box<dyn DynIncompatible> { //~ ERROR the trait `DynIncompatible` is not dyn compatible
    if true {
        return Box::new(A); //~ ERROR is not dyn compatible
    }
    Box::new(B) //~ ERROR is not dyn compatible
}

fn main() {}
