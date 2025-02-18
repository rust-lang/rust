#![allow(dead_code)]

struct Struct;
trait Trait {}
impl Trait for Struct {}
impl Trait for u32 {}

fn foo() -> dyn Trait { Struct }
//~^ ERROR return type cannot be a trait object without pointer indirection
//~| ERROR return type cannot be a trait object without pointer indirection

fn bar() -> dyn Trait {
    //~^ ERROR return type cannot be a trait object without pointer indirection
    //~| ERROR return type cannot be a trait object without pointer indirection
    if true {
        return 0;
    }
    42
}

fn main() {}
