#![feature(negative_impls)]
#![feature(marker_trait_attr)]

#[marker]
trait MyTrait {}

struct TestType<T>(::std::marker::PhantomData<T>);

unsafe impl<T: MyTrait + 'static> Send for TestType<T> {}

impl<T: MyTrait> !Send for TestType<T> {}
//~^ ERROR found both positive and negative implementation
//~| ERROR `!Send` impl requires `T: MyTrait` but the struct it is implemented for does not

unsafe impl<T: 'static> Send for TestType<T> {}
//~^ ERROR conflicting implementations

impl !Send for TestType<i32> {}
//~^ ERROR `!Send` impls cannot be specialized

fn main() {}
