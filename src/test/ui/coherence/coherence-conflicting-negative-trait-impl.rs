// revisions: old re

#![cfg_attr(re, feature(re_rebalance_coherence))]
#![feature(optin_builtin_traits)]
#![feature(overlapping_marker_traits)]

trait MyTrait {}

struct TestType<T>(::std::marker::PhantomData<T>);

unsafe impl<T: MyTrait+'static> Send for TestType<T> {}

impl<T: MyTrait> !Send for TestType<T> {}
//[old]~^ ERROR conflicting implementations of trait `std::marker::Send`
//[re]~^^ ERROR E0119

unsafe impl<T:'static> Send for TestType<T> {}

impl !Send for TestType<i32> {}
//[old]~^ ERROR conflicting implementations of trait `std::marker::Send`
//[re]~^^ ERROR E0119

fn main() {}
