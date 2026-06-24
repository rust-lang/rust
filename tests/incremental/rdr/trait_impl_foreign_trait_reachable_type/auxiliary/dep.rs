//@ compile-flags: -Z public-api-hash

#![crate_name = "dep"]
#![crate_type = "rlib"]

pub struct S;

pub fn anchor() {}

#[cfg(any(cpass2))]
impl std::fmt::Display for S {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "S")
    }
}
