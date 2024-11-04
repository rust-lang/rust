pub trait Argument {}
impl Argument for u8 {}
impl Argument for i8 {}
impl Argument for String {}
impl Argument for &str {}

pub trait TupleArgs {}
impl<A: Argument> TupleArgs for (A,) {}
impl<A: Argument, B: Argument> TupleArgs for (A, B) {}
impl<A: Argument, B: Argument, C: Argument> TupleArgs for (A, B, C) {}

fn convert_into_tuple(_x: impl TupleArgs) {}

fn main() {
    convert_into_tuple(42_u8);
    //~^ ERROR E0277
    //~| HELP the following other types implement trait `TupleArgs`
    //~| HELP use a unary tuple instead
}
