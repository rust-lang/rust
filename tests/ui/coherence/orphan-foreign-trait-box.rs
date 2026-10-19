//! Regression test for #164592: the primary span and labels of E0117
//! must not overlap.

impl Into<i32> for Box<i32> {}
//~^ ERROR only traits defined in the current crate can be implemented for types defined outside of the crate

fn main() {}
