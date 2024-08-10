#![crate_name = "dependency"]
#![crate_type = "rlib"]
pub struct Type;
pub trait Trait {
    fn foo(&self);
    fn bar();
}
impl Trait for Type {
    fn foo(&self) {}
    fn bar() {}
}
pub fn do_something<X: Trait>(_: X) {}
