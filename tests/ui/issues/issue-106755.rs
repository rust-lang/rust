// compile-flags:-Ztranslate-lang=en_US

#![feature(negative_impls)]
#![feature(marker_trait_attr)]

#[marker]
trait MyTrait {}

struct TestType<T>(::std::marker::PhantomData<T>);

unsafe impl<T: MyTrait + 'static> Send for TestType<T> {}

impl<T: MyTrait> !Send for TestType<T> {} //~ ERROR found both positive and negative implementation

unsafe impl<T: 'static> Send for TestType<T> {} //~ ERROR conflicting implementations

impl !Send for TestType<i32> {}

fn main() {}
