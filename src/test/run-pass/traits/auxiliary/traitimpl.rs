// Test inherent trait impls work cross-crait.

pub trait Bar<'a> : 'a {}

impl<'a> Bar<'a> {
    pub fn bar(&self) {}
}
