//@ compile-flags: -Z public-api-hash

#![crate_name = "dep"]
#![crate_type = "rlib"]
#![allow(dead_code)]

// Private type: nothing public references it, so it is not reachable and downstream
// crates cannot name it.
struct Private;

pub fn anchor() {}

#[cfg(any(cpass2))]
impl std::fmt::Display for Private {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "p")
    }
}
