#![crate_name = "dependency"]
#![crate_type = "rlib"]
pub struct Type;
pub trait Trait {
    fn foo(&self);
    fn bar();
}
pub trait Trait2 {}
impl Trait2 for Type {}
impl Trait for Type {
    fn foo(&self) {}
    fn bar() {}
}
pub fn do_something<X: Trait>(_: X) {}
pub fn do_something_type(_: Type) {}
pub fn do_something_trait(_: Box<dyn Trait2>) {}

#[derive(Debug)]
pub struct Error;

impl From<()> for Error {
    fn from(t: ()) -> Error {
        Error
    }
}

#[derive(Debug)]
pub struct OtherError;

impl From<()> for OtherError {
    fn from(_: ()) -> OtherError {
        OtherError
    }
}

impl From<i32> for OtherError {
    fn from(_: i32) -> OtherError {
        OtherError
    }
}
