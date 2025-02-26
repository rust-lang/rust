#![crate_type = "lib"]
#![crate_name = "foo"]

extern crate foo;

pub struct Struct;
pub trait Trait {}
impl Trait for Struct {}

fn check_trait<T: Trait>() {}

fn ice() {
    check_trait::<foo::Struct>();
}
