// Test that we DO NOT warn when lifetime name is used only
// once in a fn return type -- using `'_` is not legal there,
// as it must refer back to an argument.
//
// (Normally, using `'static` would be preferred, but there are
// times when that is not what you want.)

//@ check-pass

#![deny(single_use_lifetimes)]

// OK: used only in return type
fn b<'a>() -> &'a u32 {
    &22
}

pub trait Tfv<'a> {}
impl Tfv<'_> for () {}

// Do NOT lint if used in return type.
pub fn i<'a>() -> impl Tfv<'a> {}

fn main() {}
