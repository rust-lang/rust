//#![feature(non_exhaustive)]

#[non_exhaustive] //~ERROR non exhaustive is an experimental feature
pub enum NonExhaustiveEnum {
    Unit,
    Tuple(u32),
    Struct { field: u32 }
}

fn main() { }
