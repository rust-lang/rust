#![crate_name = "foo"]

pub struct Struct;

pub trait Tr1 {}
pub trait Tr2 {
    type Assoc;
}
impl Tr2 for () {
    type Assoc = Struct;
}

impl Tr1 for <() as Tr2>::Assoc {}
