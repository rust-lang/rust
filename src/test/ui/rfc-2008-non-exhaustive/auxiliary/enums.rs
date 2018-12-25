#![crate_type = "rlib"]
#![feature(non_exhaustive)]

#[non_exhaustive]
pub enum NonExhaustiveEnum {
    Unit,
    Tuple(u32),
    Struct { field: u32 }
}
