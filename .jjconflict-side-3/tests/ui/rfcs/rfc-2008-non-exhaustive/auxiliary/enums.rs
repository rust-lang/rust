#![crate_type = "rlib"]

#[non_exhaustive]
pub enum NonExhaustiveEnum {
    Unit,
    Tuple(u32),
    Struct { field: u32 },
}

#[non_exhaustive]
pub enum NestedNonExhaustive {
    A(NonExhaustiveEnum),
    B,
    C,
}

#[non_exhaustive]
pub enum EmptyNonExhaustiveEnum {}

pub enum VariantNonExhaustive {
    #[non_exhaustive]
    Bar {
        x: u32,
        y: u64,
    },
    Baz(u32, u16),
}

#[non_exhaustive]
pub enum NonExhaustiveSingleVariant {
    A(bool),
}

#[repr(u8)]
pub enum FieldLessWithNonExhaustiveVariant {
    A,
    B,
    #[non_exhaustive]
    C,
}

impl Default for FieldLessWithNonExhaustiveVariant {
    fn default() -> Self { Self::A }
}
