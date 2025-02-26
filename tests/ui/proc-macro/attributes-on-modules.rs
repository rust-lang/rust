//@ check-pass
//@ proc-macro: test-macros.rs

#[macro_use]
extern crate test_macros;

#[identity_attr]
mod m {
    pub struct S;
}

#[identity_attr]
fn f() {
    mod m {}
}

fn main() {
    let s = m::S;
}
