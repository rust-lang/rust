//@ known-bug: #111742
// ignore-tidy-linelength

#![allow(incomplete_features)]
#![feature(generic_const_exprs)]

const CONST: u32 = 0;
struct Test<const N: u32, const M: u32 = { CONST/* Must be a const and not a Literal */ }> where [(); N as usize]: , ([u32; N as usize]);

fn main() {
    let _: Test<1>;
}
