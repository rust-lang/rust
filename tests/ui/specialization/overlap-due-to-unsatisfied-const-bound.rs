// Regression test for #140571. The compiler used to ICE

#![feature(min_generic_const_args, specialization)]

pub trait IsVoid {
    #[rustc_always_gca]
    const IS_VOID: bool;
}
impl<T> IsVoid for T {
    default const IS_VOID: bool = core::direct_const_arg!(false);
}

pub trait NotVoid {}
impl<T> NotVoid for T where T: IsVoid<IS_VOID = false> + ?Sized {}

pub trait Maybe<T> {}
impl<T> Maybe<T> for T {}
impl<T> Maybe<T> for () where T: NotVoid + ?Sized {}
//~^ ERROR conflicting implementations of trait `Maybe<()>` for type `()`

fn main() {}
