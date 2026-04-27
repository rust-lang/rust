// Test that we check the body at the definition site for well-formedness.

#![feature(lazy_type_alias)]

// unsatisified trait bounds
type _A<T> = <T as std::ops::Mul>::Output; //~ ERROR cannot multiply `T` by `T`
type _B = Vec<str>; //~ ERROR the size for values of type `str` cannot be known at compilation time

// unsatisfied outlives-bounds
type _C<'a> = &'static &'a (); //~ ERROR reference has a longer lifetime than the data it references

// diverging const exprs
type _D = [(); panic!()]; //~ ERROR evaluation panicked

// dyn incompatibility
type _E = dyn Sized; //~ ERROR the trait `Sized` is not dyn compatible

fn main() {}
