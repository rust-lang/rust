#![crate_name = "dependency"]
#![crate_type = "rlib"]
pub struct Type;
pub trait Trait {}
impl Trait for Type {}
pub fn do_something<X: Trait>(_: X) {}
