//! Regression test for #155168
//!
//! Ensure that providing an array const arg with the wrong number of elements
//! doesn't ICE or silently cause UB.
#![expect(incomplete_features)]
#![feature(adt_const_params, min_generic_const_args, macroless_generic_const_args)]
#![feature(unsized_const_params, generic_const_parameter_types)]

use std::gca;
use std::marker::ConstParamTy_;

fn foo<T: ConstParamTy_, const N: usize, const M: [T; N]>() -> [T; N] {
    M
}

fn bar<const A: [u8; 2]>() {}

trait Trait {
    #[rustc_always_gca]
    const LEN: usize;
}

struct S;
impl Trait for S {
    const LEN: usize = gca!(3);
}

fn baz<const A: [u8; <S as Trait>::LEN]>() {}

fn main() {
    foo::<u8, 2, { [] }>();
    //~^ ERROR: the constant `*b""` is not of type `[u8; 2]`
    foo::<u8, 2, { [0, 0, 0] }>();
    //~^ ERROR: the constant `*b"\x00\x00\x00"` is not of type `[u8; 2]`
    bar::<{ [] }>();
    //~^ ERROR: the constant `*b""` is not of type `[u8; 2]`
    bar::<{ [1, 2, 3] }>();
    //~^ ERROR: the constant `*b"\x01\x02\x03"` is not of type `[u8; 2]`
    baz::<{ [42] }>();
    //~^ ERROR: the constant `*b"*"` is not of type `[u8; 3]`
}
