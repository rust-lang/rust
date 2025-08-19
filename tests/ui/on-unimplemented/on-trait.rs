// ignore-tidy-linelength

#![feature(rustc_attrs)]

pub mod Bar {
  #[rustc_on_unimplemented = "test error `{Self}` with `{Bar}` `{Baz}` `{Quux}` in `{This}`"]
  pub trait Foo<Bar, Baz, Quux> {}
}

use Bar::Foo;

fn foobar<U: Clone, T: Foo<u8, U, u32>>() -> T {
    panic!()
}

#[rustc_on_unimplemented="a collection of type `{Self}` cannot be built from an iterator over elements of type `{A}`"]
trait MyFromIterator<A> {
    /// Builds a container with elements from an external iterator.
    fn my_from_iter<T: Iterator<Item=A>>(iterator: T) -> Self;
}

fn collect<A, I: Iterator<Item=A>, B: MyFromIterator<A>>(it: I) -> B {
    MyFromIterator::my_from_iter(it)
}

pub fn main() {
    let x = vec![1u8, 2, 3, 4];
    let y: Option<Vec<u8>> = collect(x.iter()); // this should give approximately the same error for x.iter().collect()
    //~^ ERROR

    let x: String = foobar(); //~ ERROR
}
