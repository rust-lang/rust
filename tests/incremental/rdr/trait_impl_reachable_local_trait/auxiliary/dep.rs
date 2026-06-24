//@ compile-flags: -Z public-api-hash

#![crate_name = "dep"]
#![crate_type = "rlib"]

pub trait T {
    fn m(&self);
}
pub struct S;

pub fn anchor() {}

#[cfg(any(cpass2))]
impl T for S {
    fn m(&self) {}
}
