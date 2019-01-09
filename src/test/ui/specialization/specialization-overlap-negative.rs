#![feature(optin_builtin_traits)]
#![feature(specialization)]

trait MyTrait {}

struct TestType<T>(::std::marker::PhantomData<T>);

unsafe impl<T: Clone> Send for TestType<T> {}
impl<T: MyTrait> !Send for TestType<T> {} //~ ERROR E0119

fn main() {}
