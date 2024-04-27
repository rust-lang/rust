#![crate_type="lib"]


pub trait A {
    fn a(&self) {}
}
impl A for () {}
