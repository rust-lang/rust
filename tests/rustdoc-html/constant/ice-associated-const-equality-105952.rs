//! Regression test for <https://github.com/rust-lang/rust/issues/105952>

#![crate_name = "foo"]
#![feature(min_generic_const_args, adt_const_params)]
#![expect(incomplete_features)]
use std::marker::ConstParamTy;

#[derive(PartialEq, Eq, ConstParamTy)]
pub enum ParseMode {
    Raw,
}
pub trait Parse {
    #[type_const]
    const PARSE_MODE: ParseMode;
}
pub trait RenderRaw {}

//@ hasraw foo/trait.RenderRaw.html 'impl'
//@ hasraw foo/trait.RenderRaw.html 'ParseMode::Raw'
impl<T: Parse<PARSE_MODE = { ParseMode::Raw }>> RenderRaw for T {}
