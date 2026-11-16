//@ compile-flags: -Znext-solver
//@ build-fail
//@ compare-output-by-lines
//@ forbid-output: internal compiler error

// Separate associated constants should produce separate errors.
use std::marker::PhantomData;
use std::mem::size_of;

trait Ob { type A; }
impl<T: ?Sized> Ob for T { type A = u8; }

struct First<T: ?Sized>(PhantomData<T>);
struct Second<T: ?Sized>(PhantomData<T>);
impl<T: ?Sized + Ob> First<T> {
    const SIZE: usize = size_of::<T::A>(); //~ ERROR unable to normalize
}
impl<T: ?Sized + Ob> Second<T> {
    const SIZE: usize = size_of::<T::A>(); //~ ERROR unable to normalize
}
type Object = dyn Ob<A = u16>;

fn main() {
    println!("{}", First::<Object>::SIZE);
    println!("{}", Second::<Object>::SIZE);
}

//@ normalize-stderr: "\n\n$" -> "\n"
