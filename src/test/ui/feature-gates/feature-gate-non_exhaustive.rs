//#![feature(non_exhaustive)]

#[non_exhaustive] //~ERROR non exhaustive is an experimental feature (see issue #44109)
pub enum NonExhaustiveEnum {
    Unit,
    Tuple(u32),
    Struct { field: u32 }
}

fn main() { }
