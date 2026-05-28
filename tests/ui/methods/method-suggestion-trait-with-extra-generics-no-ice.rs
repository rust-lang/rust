//! Regression test for #157189.
//!
//! When a method call fails to resolve, the "trait which provides `<method>` is
//! implemented but not in scope" diagnostic probes all traits for a method of the
//! same name. Here `.borrow()` matches `std::borrow::Borrow::borrow`, and `Borrow`
//! has a generic parameter (`Borrowed`) besides `Self`. Building the trait
//! reference for the diagnostic used to pass only the receiver type as the single
//! argument, which mismatched the trait's generics and ICEd in
//! `debug_assert_args_compatible`. It should just report the error.

trait Foo {
    extern "C" fn borrow(&self);
}

struct Bar;

fn main() {
    let foo: Box<dyn Fn(bool) -> usize> = Box::new(Bar);
    //~^ ERROR expected a `Fn(bool)` closure, found `Bar`
    foo.borrow();
    //~^ ERROR no method named `borrow` found
    foo.take()
    //~^ ERROR `Box<dyn Fn(bool) -> usize>` is not an iterator
}
