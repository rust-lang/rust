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

fn car() -> dyn DynIncompatible { //~ ERROR the trait `DynIncompatible` cannot be made into an object
//~^ ERROR return type cannot have an unboxed trait object
    if true {
        return A;
    }
    B
}

fn cat() -> Box<dyn DynIncompatible> { //~ ERROR the trait `DynIncompatible` cannot be made into an
    if true {
        return Box::new(A); //~ ERROR cannot be made into an object
    }
    Box::new(B) //~ ERROR cannot be made into an object
}

fn main() {}
