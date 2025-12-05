//@no-rustfix
#![allow(clippy::len_without_is_empty)]

// Check that the lint expectation is fulfilled even if the lint is allowed at the type level.
pub struct Empty;

impl Empty {
    #[expect(clippy::len_without_is_empty)]
    pub fn len(&self) -> usize {
        0
    }
}

// Check that the lint expectation is not triggered if it should not
pub struct Empty2;

impl Empty2 {
    #[expect(clippy::len_without_is_empty)] //~ ERROR: this lint expectation is unfulfilled
    pub fn len(&self) -> usize {
        0
    }

    pub fn is_empty(&self) -> bool {
        false
    }
}

fn main() {}
