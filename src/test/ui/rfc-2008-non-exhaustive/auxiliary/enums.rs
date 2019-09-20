#![crate_type = "rlib"]

#[non_exhaustive]
pub enum NonExhaustiveEnum {
    Unit,
    Tuple(u32),
    Struct { field: u32 }
}
