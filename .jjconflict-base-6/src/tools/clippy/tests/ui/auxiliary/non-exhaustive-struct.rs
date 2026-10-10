#[non_exhaustive]
#[derive(Default)]
pub struct NonExhaustiveStruct {
    pub field1: i32,
    pub field2: i32,
    _private: i32,
}

#[non_exhaustive]
#[derive(Default)]
pub struct NonExhaustiveStructNoPrivateFields {
    pub field: i32,
}
