//@ known-bug: rust-lang/rust#143506
#![feature(generic_const_exprs)]
#![feature(min_generic_const_args)]
fn foo<const N: u32>(a: [(); N as usize]) {}

const C: f32 = 1.0;

fn main() {
    foo::<C>();
}
