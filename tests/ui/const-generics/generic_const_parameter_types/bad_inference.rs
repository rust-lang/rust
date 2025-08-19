#![feature(adt_const_params, unsized_const_params, generic_const_parameter_types)]
#![allow(incomplete_features)]

use std::marker::ConstParamTy_;

fn foo<T: ConstParamTy_, const N: usize, const M: [T; N]>() -> [T; N] {
    loop {}
}

fn main() {
    // Requires inferring `T`/`N` from `12_u8` and `2` respectively.
    let a = foo::<_, _, { [12_u8; 2] }>();
    //~^ ERROR: anonymous constants with inferred types are not yet supported

    // Requires inferring `T`/`N`/`12_?i`/`_` from `[u8; 2]`
    let b: [u8; 2] = foo::<_, _, { [12; _] }>();
    //~^ ERROR: anonymous constants with inferred types are not yet supported
}
