// check-pass
#![deny(improper_ctypes)]

// This test checks that non-exhaustive types with `#[repr(C)]` are considered proper within
// the defining crate.

#[non_exhaustive]
#[repr(C)]
pub enum NonExhaustiveEnum {
    Unit,
    Tuple(u32),
    Struct { field: u32 },
}

#[non_exhaustive]
#[repr(C)]
pub struct NormalStruct {
    pub first_field: u16,
    pub second_field: u16,
}

#[non_exhaustive]
#[repr(C)]
pub struct UnitStruct;

#[non_exhaustive]
#[repr(C)]
pub struct TupleStruct(pub u16, pub u16);

#[repr(C)]
pub enum NonExhaustiveVariants {
    #[non_exhaustive]
    Unit,
    #[non_exhaustive]
    Tuple(u32),
    #[non_exhaustive]
    Struct { field: u32 },
}

extern "C" {
    // Unit structs aren't tested here because they will trigger `improper_ctypes` anyway.
    pub fn non_exhaustive_enum(_: NonExhaustiveEnum);
    pub fn non_exhaustive_normal_struct(_: NormalStruct);
    pub fn non_exhaustive_tuple_struct(_: TupleStruct);
    pub fn non_exhaustive_variant(_: NonExhaustiveVariants);
}

fn main() {}
