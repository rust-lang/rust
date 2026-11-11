// Test that we emit a correct structured suggestions for dynamically sized ("maybe unsized")
// function parameters.
// We used to emit a butchered suggestion if duplicate relaxed `Sized` bounds were present.
// issue: <https://github.com/rust-lang/rust/issues/127441>.

use std::fmt::Debug;

fn foo1<T: ?Sized>(a: T) {}
//~^ ERROR the size for values of type `T` cannot be known at compilation time

fn foo2<T: ?Sized + ?Sized>(a: T) {}
//~^ ERROR duplicate relaxed `Sized` bounds
//~| ERROR the size for values of type `T` cannot be known at compilation time

fn foo3<T: ?Sized + ?Sized + Debug>(a: T) {}
//~^ ERROR duplicate relaxed `Sized` bounds
//~| ERROR the size for values of type `T` cannot be known at compilation time

fn foo4<T: ?Sized + Debug + ?Sized >(a: T) {}
//~^ ERROR duplicate relaxed `Sized` bounds
//~| ERROR the size for values of type `T` cannot be known at compilation time

fn foo5(_: impl ?Sized) {}
//~^ ERROR the size for values of type `impl ?Sized` cannot be known at compilation time

fn foo6(_: impl ?Sized + ?Sized) {}
//~^ ERROR duplicate relaxed `Sized` bounds
//~| ERROR the size for values of type `impl ?Sized + ?Sized` cannot be known at compilation tim

fn foo7(_: impl ?Sized + ?Sized + Debug) {}
//~^ ERROR duplicate relaxed `Sized` bounds
//~| ERROR the size for values of type `impl ?Sized + ?Sized + Debug` cannot be known at compilation time

fn foo8(_: impl ?Sized + Debug + ?Sized ) {}
//~^ ERROR duplicate relaxed `Sized` bounds
//~| ERROR the size for values of type `impl ?Sized + Debug + ?Sized` cannot be known at compilation time

fn main() {}
