#![feature(adt_const_params)]
#![allow(incomplete_features)]

#[derive(PartialEq, Eq)]
struct A;

struct B<const X: A>; // ok

struct C;

struct D<const X: C>; //~ ERROR `C` must be annotated with `#[derive(PartialEq, Eq)]`

fn main() {}
