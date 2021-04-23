#![feature(associated_type_defaults)]

trait Assoc {
    fn func() {}
    const CONST: u8 = 0;
    type Ty = u8;
}

trait Dyn {}

impl Assoc for dyn Dyn {}

fn main() {
    Dyn::func(); //~ WARN trait objects without an explicit `dyn` are deprecated
    ::Dyn::func(); //~ WARN trait objects without an explicit `dyn` are deprecated
    Dyn::CONST; //~ WARN trait objects without an explicit `dyn` are deprecated
    let _: Dyn::Ty; //~ ERROR ambiguous associated type
}
