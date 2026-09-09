//@ compile-flags: -Z public-api-hash

#![crate_name = "dep"]
#![crate_type = "rlib"]
#![allow(dead_code)]

// Private trait: nothing public references it, so it is not reachable and downstream
// crates cannot import it.
trait T {
    fn m(&self);
}
pub struct S;

pub fn anchor() {}

#[cfg(any(cpass2))]
impl T for S {
    fn m(&self) {}
}
