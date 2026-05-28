//@ check-pass
// hopefully this doesn't cause an ICE

// https://github.com/rust-lang/rust/issues/46271

pub fn foo() {
    extern crate std;
}
