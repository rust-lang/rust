// run-rustfix

#[allow(unused)]
use std::fmt::Debug;
// Rustfix should add this, or use `std::fmt::Debug` instead.

#[allow(dead_code)]
fn test_impl(t: impl Sized) {
    println!("{:?}", t);
    //~^ ERROR doesn't implement
}

#[allow(dead_code)]
fn test_no_bounds<T>(t: T) {
    println!("{:?}", t);
    //~^ ERROR doesn't implement
}

#[allow(dead_code)]
fn test_one_bound<T: Sized>(t: T) {
    println!("{:?}", t);
    //~^ ERROR doesn't implement
}

#[allow(dead_code)]
fn test_no_bounds_where<X, Y>(x: X, y: Y) where X: std::fmt::Debug, {
    println!("{:?} {:?}", x, y);
    //~^ ERROR doesn't implement
}

#[allow(dead_code)]
fn test_one_bound_where<X>(x: X) where X: Sized {
    println!("{:?}", x);
    //~^ ERROR doesn't implement
}

#[allow(dead_code)]
fn test_many_bounds_where<X>(x: X) where X: Sized, X: Sized {
    println!("{:?}", x);
    //~^ ERROR doesn't implement
}

trait Foo<T> {
    const SIZE: usize = core::mem::size_of::<Self>();
    //~^ ERROR the size for values of type `Self` cannot be known at compilation time
}

trait Bar: std::fmt::Display {
    const SIZE: usize = core::mem::size_of::<Self>();
    //~^ ERROR the size for values of type `Self` cannot be known at compilation time
}

trait Baz where Self: std::fmt::Display {
    const SIZE: usize = core::mem::size_of::<Self>();
    //~^ ERROR the size for values of type `Self` cannot be known at compilation time
}

trait Qux<T> where Self: std::fmt::Display {
    const SIZE: usize = core::mem::size_of::<Self>();
    //~^ ERROR the size for values of type `Self` cannot be known at compilation time
}

trait Bat<T>: std::fmt::Display {
    const SIZE: usize = core::mem::size_of::<Self>();
    //~^ ERROR the size for values of type `Self` cannot be known at compilation time
}

fn main() { }
