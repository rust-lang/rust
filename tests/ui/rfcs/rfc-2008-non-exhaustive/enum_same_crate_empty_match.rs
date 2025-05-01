//@ dont-require-annotations: NOTE

#![deny(unreachable_patterns)]

#[non_exhaustive]
pub enum NonExhaustiveEnum {
    Unit,
    //~^ NOTE not covered
    Tuple(u32),
    //~^ NOTE not covered
    Struct { field: u32 }
    //~^ NOTE not covered
}

pub enum NormalEnum {
    Unit,
    //~^ NOTE not covered
    Tuple(u32),
    //~^ NOTE not covered
    Struct { field: u32 }
    //~^ NOTE not covered
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
    //~^ ERROR `NonExhaustiveEnum::Unit`, `NonExhaustiveEnum::Tuple(_)` and `NonExhaustiveEnum::Struct { .. }` not covered [E0004]
    match NormalEnum::Unit {}
    //~^ ERROR `NormalEnum::Unit`, `NormalEnum::Tuple(_)` and `NormalEnum::Struct { .. }` not covered [E0004]
}
