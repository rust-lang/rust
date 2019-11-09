#![feature(optin_builtin_traits)]
#![feature(overlapping_marker_traits)]

trait MyTrait {}

struct TestType<T>(::std::marker::PhantomData<T>);

unsafe impl<T: MyTrait+'static> Send for TestType<T> {}

impl<T: MyTrait> !Send for TestType<T> {}
//~^ ERROR E0119

unsafe impl<T:'static> Send for TestType<T> {}

impl !Send for TestType<i32> {}
//~^ ERROR E0119

fn main() {}
