// check-pass
// compile-flags: -Zmir-opt-level=3 --crate-type=lib

pub trait Factory<T> {
    type Item;
}

pub struct IntFactory;

impl<T> Factory<T> for IntFactory {
    type Item = usize;
}

pub fn foo<T>() where IntFactory: Factory<T> {
    let mut x: <IntFactory as Factory<T>>::Item = bar::<T>();
}

#[inline]
pub fn bar<T>() -> <IntFactory as Factory<T>>::Item {
    0usize
}
