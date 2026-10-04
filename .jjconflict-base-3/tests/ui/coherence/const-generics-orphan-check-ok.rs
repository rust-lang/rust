//@ check-pass
//@ aux-build:trait-with-const-param.rs
extern crate trait_with_const_param;
use trait_with_const_param::*;

// Trivial case, const param after local type.
struct Local1;
impl<const N: usize, T> Trait<N, T> for Local1 {}

// Concrete consts behave the same as foreign types,
// so this also trivially works.
impl Trait<3, Local1> for i32 {}

// This case isn't as trivial as we would forbid type
// parameters here, we do allow const parameters though.
//
// The reason that type parameters are forbidden for
// `impl<T> Trait<T, LocalInA> for i32 {}` is that another
// downstream crate can add `impl<T> Trait<LocalInB, T> for i32`.
// As these two impls would overlap we forbid any impls which
// have a type parameter in front of a local type.
//
// With const parameters this issue does not exist as there are no
// constants local to another downstream crate.
struct Local2;
impl<const N: usize> Trait<N, Local2> for i32 {}

fn main() {}
