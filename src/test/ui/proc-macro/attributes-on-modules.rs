// aux-build:test-macros.rs

#[macro_use]
extern crate test_macros;

#[identity_attr] //~ ERROR custom attributes cannot be applied to modules
mod m {
    pub struct S;
}

fn main() {
    let s = m::S;
}
