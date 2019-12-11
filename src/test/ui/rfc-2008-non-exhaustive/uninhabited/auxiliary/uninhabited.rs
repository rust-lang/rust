#![crate_type = "rlib"]
#![feature(never_type)]

#[non_exhaustive]
pub enum UninhabitedEnum {
}

#[non_exhaustive]
pub struct UninhabitedStruct {
    _priv: !,
}

#[non_exhaustive]
pub struct UninhabitedTupleStruct(!);

pub enum UninhabitedVariants {
    #[non_exhaustive] Tuple(!),
    #[non_exhaustive] Struct { x: ! }
}

pub enum PartiallyInhabitedVariants {
    Tuple(u8),
    #[non_exhaustive] Struct { x: ! }
}

pub struct IndirectUninhabitedEnum(UninhabitedEnum);

pub struct IndirectUninhabitedStruct(UninhabitedStruct);

pub struct IndirectUninhabitedTupleStruct(UninhabitedTupleStruct);

pub struct IndirectUninhabitedVariants(UninhabitedVariants);
