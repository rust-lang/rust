//! Regression test for <https://github.com/rust-lang/rust/issues/152683>
#![feature(
    adt_const_params,
    gca_adts,
    gca_macroless_args,
    gca_min_const_items,
    generic_const_parameter_types
)]
fn foo<const N: usize, const A: [u8; N]>() {}

fn main() {
    foo::<_, { [0, 1u8, 2u32, 8u64] }>();
    //~^ ERROR the constant `2` is not of type `u8`
    //~| ERROR the constant `8` is not of type `u8`
}
