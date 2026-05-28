//! Regression test for <https://github.com/rust-lang/rust/issues/152683>
#![expect(incomplete_features)]
#![feature(adt_const_params, generic_const_parameter_types, min_generic_const_args)]
fn foo<const N: usize, const A: [u8; N]>() {}

fn main() {
    foo::<_, { [0, 1u8, 2u32, 8u64] }>();
    //~^ ERROR the constant `2` is not of type `u8`
    //~| ERROR the constant `8` is not of type `u8`
}
