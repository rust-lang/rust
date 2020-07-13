#![feature(negative_impls)]
#![feature(specialization)] //~ WARN the feature `specialization` is incomplete

trait MyTrait {}

struct TestType<T>(::std::marker::PhantomData<T>);

unsafe impl<T: Clone> Send for TestType<T> {}
impl<T: MyTrait> !Send for TestType<T> {} //~ ERROR E0751

fn main() {}
