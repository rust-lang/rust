pub struct Abc;

pub trait Bcd {}

pub struct Def<'a> {
    pub field1: Abc,
    pub field2: &'a dyn Bcd,
}
