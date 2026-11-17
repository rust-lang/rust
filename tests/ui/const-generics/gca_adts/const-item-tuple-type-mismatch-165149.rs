//! Regression test for <https://github.com/rust-lang/rust/issues/165149>.
#![feature(gca_macroless_items, gca_adts, min_adt_const_params, gca_min_const_items)]

const INTS: (usize, u8) = ("a", true);
//~^ ERROR the constant `"a"` is not of type `usize`
//~| ERROR the constant `true` is not of type `u8`

const STRS: (&'static str, &'static str) = ("a", true);
//~^ ERROR the constant `true` is not of type `&'static str`

fn main() {
    INTS;
    STRS;
}
