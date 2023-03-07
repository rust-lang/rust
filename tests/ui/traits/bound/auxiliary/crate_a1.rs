pub trait Bar {}

pub fn try_foo(x: impl Bar) {}

pub struct ImplementsTraitForUsize<T> {
    _marker: std::marker::PhantomData<T>,
}

impl Bar for ImplementsTraitForUsize<usize> {}
