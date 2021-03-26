pub struct Foo;

pub trait Bar {}

impl Bar for Foo {}

pub struct DoesNotImplementTrait;

pub struct ImplementsWrongTraitConditionally<T> {
    _marker: std::marker::PhantomData<T>,
}

impl Bar for ImplementsWrongTraitConditionally<isize> {}
