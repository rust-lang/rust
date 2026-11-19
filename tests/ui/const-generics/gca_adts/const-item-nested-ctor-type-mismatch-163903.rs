//! Regression test for <https://github.com/rust-lang/rust/issues/163903>.
#![feature(gca_min_const_items, gca_macroless_items, gca_adts, min_adt_const_params)]

use std::marker::ConstParamTy;

#[derive(Eq, PartialEq)]
struct NoDerive(i32);

#[derive(ConstParamTy, PartialEq, Eq)]
struct WrapInline(NoDerive);
//~^ ERROR the trait `ConstParamTy_` cannot be implemented for this type

const WRAP_DIRECT_INLINE: WrapInline = WrapInline(NoDerive(NoDerive));
//~^ ERROR the constant `NoDerive` is not of type `i32`

fn main() {
    match WRAP_DIRECT_INLINE {
        _ => {}
    }
}
