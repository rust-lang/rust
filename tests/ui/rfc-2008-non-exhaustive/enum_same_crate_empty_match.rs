#![deny(unreachable_patterns)]

#[non_exhaustive]
pub enum NonExhaustiveEnum {
    Unit,
    //~^ not covered
    Tuple(u32),
    //~^ not covered
    Struct { field: u32 }
    //~^ not covered
}

pub enum NormalEnum {
    Unit,
    //~^ not covered
    Tuple(u32),
    //~^ not covered
    Struct { field: u32 }
    //~^ not covered
}

#[non_exhaustive]
pub enum EmptyNonExhaustiveEnum {}

fn empty_non_exhaustive(x: EmptyNonExhaustiveEnum) {
    match x {}
    match x {
        _ => {} //~ ERROR unreachable pattern
    }
}

fn main() {
    match NonExhaustiveEnum::Unit {}
    //~^ ERROR match is non-exhaustive [E0004]
    match NormalEnum::Unit {}
    //~^ ERROR match is non-exhaustive [E0004]
}
