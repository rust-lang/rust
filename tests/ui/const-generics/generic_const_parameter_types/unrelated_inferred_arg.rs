//@ check-pass

#![feature(adt_const_params, unsized_const_params, generic_const_parameter_types)]
#![allow(incomplete_features)]

use std::marker::ConstParamTy_;

fn foo<U, T: ConstParamTy_, const N: usize, const M: [T; N]>(_: U) -> [T; N] {
    loop {}
}

fn main() {
    // Check that `_` doesnt cause a "Type of const argument is uninferred" error
    // as it is not actually used by the type of `M`.
    let a = foo::<_, u8, 2, { [12; _] }>(true);
}
