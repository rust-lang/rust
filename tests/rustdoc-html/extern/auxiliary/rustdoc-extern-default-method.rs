#![crate_type="lib"]

pub trait Trait {
    fn provided(&self) {}
}

pub struct Struct;

impl Trait for Struct {
    fn provided(&self) {}
}
