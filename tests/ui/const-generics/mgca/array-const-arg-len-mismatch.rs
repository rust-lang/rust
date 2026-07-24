//! Regression test for #155168
//!
//! Ensure that providing an array const arg with the wrong number of elements
//! doesn't ICE or silently cause UB.
#![expect(incomplete_features)]
#![feature(adt_const_params, min_generic_const_args)]
#![feature(unsized_const_params, generic_const_parameter_types)]

use std::marker::ConstParamTy_;

fn foo<T: ConstParamTy_, const N: usize, const M: [T; N]>() -> [T; N] {
    M
}

fn bar<const A: [u8; 2]>() {}

trait Trait {
    type const LEN: usize;
}

struct S;
impl Trait for S {
    type const LEN: usize = 3;
}

fn baz<const A: [u8; <S as Trait>::LEN]>() {}

fn main() {
    foo::<u8, 2, { [] }>();
    //~^ ERROR: expected array with 2 elements, found 0 elements
    foo::<u8, 2, { [0, 0, 0] }>();
    //~^ ERROR: expected array with 2 elements, found 3 elements
    bar::<{ [] }>();
    //~^ ERROR: expected array with 2 elements, found 0 elements
    bar::<{ [1, 2, 3] }>();
    //~^ ERROR: expected array with 2 elements, found 3 elements
    baz::<{ [42] }>();
    //~^ ERROR: expected array with 3 elements, found 1 elements
}
