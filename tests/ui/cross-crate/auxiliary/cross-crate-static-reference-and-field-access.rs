//! Auxiliary crate testing this issue https://github.com/rust-lang/rust/issues/29265
#![crate_type = "lib"]

pub struct SomeType {
    pub some_member: usize,
}

pub static SOME_VALUE: SomeType = SomeType {
    some_member: 1,
};
