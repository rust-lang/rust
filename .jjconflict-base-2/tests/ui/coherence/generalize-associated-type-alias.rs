// Regression test for https://github.com/rust-lang/rust/issues/154189.
#![feature(unboxed_closures)]

trait ToUnit<'a> {
    type Unit;
}

impl ToUnit<'_> for *const u32 {
    type Unit = ();
}

trait Overlap<T> {}

type Assoc<'a, T> = <*const T as ToUnit<'a>>::Unit;

impl<T> Overlap<T> for T {}

impl<'a, T> Overlap<(&'a (), Assoc<'a, T>)> for T {}
//~^ ERROR the trait bound `*const T: ToUnit<'a>` is not satisfied
//~| ERROR the trait bound `T: Overlap<(&'a (), _)>` is not satisfied

fn main() {}
