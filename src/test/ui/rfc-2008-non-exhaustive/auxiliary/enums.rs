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
