#![crate_name = "foo"]
#![crate_type = "rlib"]

extern crate dependency;
pub use dependency::{Error, OtherError, Trait2, Type, do_something_trait, do_something_type};
pub struct OtherType;
impl dependency::Trait for OtherType {
    fn foo(&self) {}
    fn bar() {}
}
#[derive(Debug)]
pub struct Error2;

impl From<Error> for Error2 {
    fn from(_: Error) -> Error2 {
        Error2
    }
}

impl From<OtherError> for Error2 {
    fn from(_: OtherError) -> Error2 {
        Error2
    }
}
