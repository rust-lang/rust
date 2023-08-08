#![allow(dead_code)]

struct Struct;
trait Trait {}
impl Trait for Struct {}
impl Trait for u32 {}

fn foo() -> dyn Trait { Struct }
//~^ ERROR E0746

fn bar() -> dyn Trait { //~ ERROR E0746
    if true {
        return 0;
    }
    42
}

fn main() {}
