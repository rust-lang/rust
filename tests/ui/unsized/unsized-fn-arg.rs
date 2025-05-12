//@ run-rustfix
#![crate_type="lib"]
#![allow(unused)]

fn f<T: ?Sized>(t: T) {}
//~^ ERROR the size for values of type `T` cannot be known at compilation time
