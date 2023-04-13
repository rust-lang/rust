// Test inherent trait impls work cross-crate.

pub trait Bar<'a> : 'a {}

impl<'a> Bar<'a> {
    pub fn bar(&self) {}
}
