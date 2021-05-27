#![feature(associated_type_defaults)]

trait Assoc {
    fn func() {}
    const CONST: u8 = 0;
    type Ty = u8;
}

trait Dyn {}

impl Assoc for dyn Dyn {}

fn main() {
    Dyn::func();
    //~^ WARN trait objects without an explicit `dyn` are deprecated
    //~| WARN this was previously accepted by the compiler
    ::Dyn::func();
    //~^ WARN trait objects without an explicit `dyn` are deprecated
    //~| WARN this was previously accepted by the compiler
    Dyn::CONST;
    //~^ WARN trait objects without an explicit `dyn` are deprecated
    //~| WARN this was previously accepted by the compiler
    let _: Dyn::Ty; //~ ERROR ambiguous associated type
}
