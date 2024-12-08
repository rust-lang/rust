#[derive(Default)]
#[non_exhaustive]
pub struct NormalStruct {
    pub first_field: u16,
    pub second_field: u16,
}

#[non_exhaustive]
pub struct UnitStruct;

#[non_exhaustive]
pub struct TupleStruct(pub u16, pub u16);

#[derive(Debug)]
#[non_exhaustive]
pub struct FunctionalRecord {
    pub first_field: u16,
    pub second_field: u16,
    pub third_field: bool,
}

impl Default for FunctionalRecord {
    fn default() -> FunctionalRecord {
        FunctionalRecord { first_field: 640, second_field: 480, third_field: false }
    }
}

#[derive(Default)]
#[non_exhaustive]
pub struct NestedStruct {
    pub foo: u16,
    pub bar: NormalStruct,
}

#[derive(Default)]
#[non_exhaustive]
pub struct MixedVisFields {
    pub a: u16,
    pub b: bool,
    pub(crate) foo: bool,
}
