#![feature(optin_builtin_traits)]

use std::marker::Send;

struct TestType;

impl !TestType {}
//~^ ERROR inherent impls cannot be negative

trait TestTrait {}

unsafe impl !Send for TestType {}
//~^ ERROR negative impls cannot be unsafe
impl !TestTrait for TestType {}
//~^ ERROR negative impls are only allowed for auto traits

struct TestType2<T>(T);

impl<T> !TestType2<T> {}
//~^ ERROR inherent impls cannot be negative

unsafe impl<T> !Send for TestType2<T> {}
//~^ ERROR negative impls cannot be unsafe
impl<T> !TestTrait for TestType2<T> {}
//~^ ERROR negative impls are only allowed for auto traits

fn main() {}
