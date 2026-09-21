//@ check-pass

#![feature(min_adt_const_params, min_generic_const_args, generic_const_parameter_types)]
fn foo<const N: usize, const A: [u8; N]>() {}

fn main() {
    foo::<_, core::direct_const_arg!([0, 1, 2, 3])>();
}
