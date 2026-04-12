//@ known-bug: rust-lang/rust#146384
#![feature(generic_const_exprs)]

const fn make_tuple<const N: usize>() -> (u8::N) {
}
type TupleConst<const N: usize> = typeof(make_tuple::<N>());
