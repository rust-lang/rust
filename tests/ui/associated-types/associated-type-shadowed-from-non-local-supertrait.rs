// Test that no help message is emitted that suggests renaming the
// associated type from a non-local trait

pub trait NewIter: Iterator {
    type Item;
}

impl<T> Clone for Box<dyn NewIter<Item = T>> {
    //~^ ERROR the value of the associated type `Item` in `Iterator` must be specified
    fn clone(&self) -> Self {
        unimplemented!();
    }
}

pub fn main() {}
