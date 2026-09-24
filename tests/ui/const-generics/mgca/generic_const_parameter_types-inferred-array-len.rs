//@ check-pass

#![feature(min_adt_const_params, min_generic_const_args, generic_const_parameter_types)]

use std::gca;

fn foo<const N: usize, const A: [u8; N]>() {}

fn main() {
    foo::<_, gca!([0, 1, 2, 3])>();
}
