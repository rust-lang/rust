#[non_exhaustive]
#[repr(C)]
pub enum NonExhaustiveEnum {
    Unit,
    Tuple(u32),
    Struct { field: u32 }
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
pub struct TupleStruct (pub u16, pub u16);

#[repr(C)]
pub enum NonExhaustiveVariants {
    #[non_exhaustive] Unit,
    #[non_exhaustive] Tuple(u32),
    #[non_exhaustive] Struct { field: u32 }
}

// Note the absence of repr(C): it's not necessary, and recent C code can now use repr hints too.
#[repr(u32)]
#[non_exhaustive]
pub enum NonExhaustiveCLikeEnum {
    One = 1,
    Two = 2,
    Three = 3,
    Four = 4,
    Five = 5,
}

#[repr(C)]
pub struct NormalStructWithNonExhaustiveCLikeEnum {
    one: u8,
    two: NonExhaustiveCLikeEnum,
}
