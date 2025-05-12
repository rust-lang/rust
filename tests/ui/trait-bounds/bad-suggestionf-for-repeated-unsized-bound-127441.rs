// Regression test for #127441

// Tests that we make the correct suggestion
// in case there are more than one `?Sized`
// bounds on a function parameter

use std::fmt::Debug;

fn foo1<T: ?Sized>(a: T) {}
//~^ ERROR he size for values of type `T` cannot be known at compilation time

fn foo2<T: ?Sized + ?Sized>(a: T) {}
//~^ ERROR type parameter has more than one relaxed default bound, only one is supported
//~| ERROR the size for values of type `T` cannot be known at compilation time

fn foo3<T: ?Sized + ?Sized + Debug>(a: T) {}
//~^ ERROR type parameter has more than one relaxed default bound, only one is supported
//~| ERROR he size for values of type `T` cannot be known at compilation time

fn foo4<T: ?Sized + Debug + ?Sized >(a: T) {}
//~^ ERROR type parameter has more than one relaxed default bound, only one is supported
//~| ERROR the size for values of type `T` cannot be known at compilation time

fn foo5(_: impl ?Sized) {}
//~^ ERROR the size for values of type `impl ?Sized` cannot be known at compilation time

fn foo6(_: impl ?Sized + ?Sized) {}
//~^ ERROR type parameter has more than one relaxed default bound, only one is supported
//~| ERROR the size for values of type `impl ?Sized + ?Sized` cannot be known at compilation tim

fn foo7(_: impl ?Sized + ?Sized + Debug) {}
//~^ ERROR type parameter has more than one relaxed default bound, only one is supported
//~| ERROR the size for values of type `impl ?Sized + ?Sized + Debug` cannot be known at compilation time

fn foo8(_: impl ?Sized + Debug + ?Sized ) {}
//~^ ERROR type parameter has more than one relaxed default bound, only one is supported
//~| ERROR the size for values of type `impl ?Sized + Debug + ?Sized` cannot be known at compilation time

fn main() {}
