#![crate_name = "foo"]
#![crate_type = "rlib"]

extern crate dependency;
pub use dependency::{Trait2, Type, do_something_trait, do_something_type};
pub struct OtherType;
impl dependency::Trait for OtherType {
    fn foo(&self) {}
    fn bar() {}
}
