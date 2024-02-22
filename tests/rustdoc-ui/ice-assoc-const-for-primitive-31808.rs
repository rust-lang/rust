//@ check-pass

// Test that associated item impls on primitive types don't crash rustdoc

// https://github.com/rust-lang/rust/issues/31808
#![crate_name="issue_31808"]

pub trait Foo {
    const BAR: usize;
    type BAZ;
}

impl Foo for () {
    const BAR: usize = 0;
    type BAZ = usize;
}
