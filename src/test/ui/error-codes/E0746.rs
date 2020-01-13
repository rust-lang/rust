struct Struct;
trait Trait {}
impl Trait for Struct {}
impl Trait for u32 {}

fn foo() -> dyn Trait { Struct }
//~^ ERROR E0746
//~| ERROR E0308

fn bar() -> dyn Trait { //~ ERROR E0746
    if true {
        return 0; //~ ERROR E0308
    }
    42 //~ ERROR E0308
}

fn main() {}
