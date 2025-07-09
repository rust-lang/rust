//@ check-pass
//@ compile-flags: --crate-type=lib
#![feature(rustc_attrs)]

#[rustc_scalable_vector(16)]
struct ScalableU8(u8);

fn main() {
    let x: [ScalableU8; 4] = todo!();
}
