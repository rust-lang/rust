//! Regression test for #163904 (used to ICE)
#![feature(gca_min_const_items, gca_macroless_items)]

const X4: &[u8; usize::MAX] = b"";
//~^ ERROR the constant `&*b""` is not of type `&'static [u8; usize::MAX]`
static Y4: u8 = X4[0];

fn main() {}
