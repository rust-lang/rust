#![crate_name = "foo"]
#![crate_type = "rlib"]

extern crate dependency;
pub use dependency::Type;
pub struct OtherType;
impl dependency::Trait for OtherType {
    fn foo(&self) {}
    fn bar() {}
}
