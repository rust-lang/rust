#![crate_type = "rlib"]
#![feature(non_exhaustive)]

pub enum NonExhaustiveVariants {
    #[non_exhaustive] Unit,
    #[non_exhaustive] Tuple(u32),
    #[non_exhaustive] Struct { field: u32 }
}
