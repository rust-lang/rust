//! Test inner attributes (#![...]) behavior in impl blocks with cfg conditions.
//!
//! This test verifies that:
//! - Inner attributes can conditionally exclude entire impl blocks
//! - Regular attributes within impl blocks work independently
//! - Attribute parsing doesn't consume too eagerly

//@ run-pass

struct Foo;

impl Foo {
    #![cfg(false)]

    fn method(&self) -> bool {
        false
    }
}

impl Foo {
    #![cfg(not(FALSE))]

    // Check that we don't eat attributes too eagerly.
    #[cfg(false)]
    fn method(&self) -> bool {
        false
    }

    fn method(&self) -> bool {
        true
    }
}

pub fn main() {
    assert!(Foo.method());
}
