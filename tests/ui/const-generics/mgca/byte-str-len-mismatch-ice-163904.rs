//! Regression test for #163904 (used to ICE)
#![feature(gca_min_const_items, gca_macroless_items)]

const X4: &[u8; 0x2468_ACF1_3579_BDFF_DB97_530E_CA86_420] = b"";
//~^ ERROR the constant `&*b""` is not of type `&'static [u8; 18282773015276577824]`
static Y4: u8 = X4[0];

fn main() {}
