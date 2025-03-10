#![feature(negative_impls)]
#![feature(specialization)] //~ WARN the feature `specialization` is incomplete

trait MyTrait {}

struct TestType<T>(::std::marker::PhantomData<T>);

unsafe impl<T: Clone> Send for TestType<T> {}
impl<T: MyTrait> !Send for TestType<T> {}
//~^ ERROR found both positive and negative implementation of trait `Send` for type `TestType<_>`
//~| ERROR `!Send` impl requires `T: MyTrait` but the struct it is implemented for does not

fn main() {}
