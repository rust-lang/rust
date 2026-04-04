//@ check-pass
//@ compile-flags: --crate-type=lib
//@ only-aarch64
#![allow(todo_macro_calls)]
#![feature(rustc_attrs)]

#[rustc_scalable_vector(16)]
struct ScalableU8(u8);

fn main() {
    let x: [ScalableU8; 4] = todo!();
}
