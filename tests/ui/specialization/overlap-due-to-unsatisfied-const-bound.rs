// Regression test for #140571. The compiler used to ICE

#![feature(associated_const_equality, specialization)]
//~^ WARN the feature `specialization` is incomplete

pub trait IsVoid {
    const IS_VOID: bool;
}
impl<T> IsVoid for T {
    default const IS_VOID: bool = false;
}

pub trait NotVoid {}
impl<T> NotVoid for T where T: IsVoid<IS_VOID = false> + ?Sized {}

pub trait Maybe<T> {}
impl<T> Maybe<T> for T {}
impl<T> Maybe<T> for () where T: NotVoid + ?Sized {}
//~^ ERROR conflicting implementations of trait `Maybe<()>` for type `()`

fn main() {}
