#[derive(Clone, Copy, PartialEq)]
pub struct HasPrivateField {
    not_pub: (),
}

pub const HAS_PRIVATE_FIELD: HasPrivateField = HasPrivateField { not_pub: () };

#[derive(Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct HasNonExhaustiveFieldList {}

pub const HAS_NON_EXHAUSTIVE_FIELD_LIST: HasNonExhaustiveFieldList = HasNonExhaustiveFieldList {};

#[derive(Clone, Copy, PartialEq)]
pub struct HasPrivateTupleField(());

pub const HAS_PRIVATE_TUPLE_FIELD: HasPrivateTupleField = HasPrivateTupleField(());

#[derive(Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct HasNonExhaustiveTupleFieldList();

pub const HAS_NON_EXHAUSTIVE_TUPLE_FIELD_LIST: HasNonExhaustiveTupleFieldList =
    HasNonExhaustiveTupleFieldList();

pub const NESTED_CONST: (HasPrivateField,) = (HAS_PRIVATE_FIELD,);

#[derive(Clone, Copy)]
pub union MixedVisibilityUnion {
    pub zst: (),
    not_pub: u8,
}
