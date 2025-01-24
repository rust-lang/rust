#![crate_name = "dependency"]
#![crate_type = "rlib"]
pub struct Type(pub i32);
pub trait Trait {
    fn foo(&self);
    fn bar();
}
pub trait Trait2 {}
impl Trait for Type {
    fn foo(&self) {}
    fn bar() {}
}
pub fn do_something<X: Trait>(_: X) {}
pub fn do_something_type(_: Type) {}
pub fn do_something_trait(_: Box<dyn Trait2>) {}
