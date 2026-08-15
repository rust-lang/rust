//@ compile-flags: --crate-type=lib
//@ check-pass
//@ revisions: e2015 e2018
//
//@[e2018] edition:2018

// Verify that gates for the `f16` and `f128` features do not apply to user modules
// See <https://github.com/rust-lang/rust/issues/123282>

mod f16 {
    pub fn a16() {}
}

mod f128 {
    pub fn a128() {}
}

pub use f128::a128;
pub use f16::a16;
