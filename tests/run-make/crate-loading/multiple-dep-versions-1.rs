#![crate_name = "dependency"]
#![crate_type = "rlib"]
pub struct Type(pub i32);
pub trait Trait {
    fn foo(&self);
}
impl Trait for Type {
    fn foo(&self) {}
}
pub fn do_something<X: Trait>(_: X) {}
