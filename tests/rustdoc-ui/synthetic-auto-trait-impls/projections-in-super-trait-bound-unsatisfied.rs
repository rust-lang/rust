// We used to ICE here while trying to synthesize auto trait impls.
// issue: 114657

pub trait Foo {
    type FooType;
}

pub trait Bar<const A: usize>: Foo<FooType = <Self as Bar<A>>::BarType> {
    type BarType;
}

pub(crate) const B: usize = 5;

pub trait Tec: Bar<B> {}

pub struct Structure<C: Tec> { //~ ERROR the trait bound `C: Bar<5>` is not satisfied
    _field: C::BarType, //~ ERROR the trait bound `C: Bar<5>` is not satisfied
}
