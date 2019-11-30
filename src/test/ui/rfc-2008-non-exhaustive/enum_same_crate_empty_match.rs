#![deny(unreachable_patterns)]

#[non_exhaustive]
pub enum NonExhaustiveEnum {
    Unit,
    //~^ variant not covered
    Tuple(u32),
    //~^ variant not covered
    Struct { field: u32 }
    //~^ variant not covered
}

pub enum NormalEnum {
    Unit,
    //~^ variant not covered
    Tuple(u32),
    //~^ variant not covered
    Struct { field: u32 }
    //~^ variant not covered
}

#[non_exhaustive]
pub enum EmptyNonExhaustiveEnum {}

fn empty_non_exhaustive(x: EmptyNonExhaustiveEnum) {
    match x {}
    match x {
        _ => {} // FIXME: should be unreachable
    }
}

fn main() {
    match NonExhaustiveEnum::Unit {}
    //~^ ERROR multiple patterns of type `NonExhaustiveEnum` are not handled [E0004]
    match NormalEnum::Unit {}
    //~^ ERROR multiple patterns of type `NormalEnum` are not handled [E0004]
}
