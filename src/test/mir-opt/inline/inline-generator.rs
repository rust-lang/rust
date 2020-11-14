// ignore-wasm32-bare compiled with panic=abort by default
#![feature(generators, generator_trait)]

use std::ops::Generator;
use std::pin::Pin;

// EMIT_MIR inline_generator.main.Inline.diff
fn main() {
    let _r = Pin::new(&mut g()).resume(false);
}

#[inline(always)]
pub fn g() -> impl Generator<bool> {
    #[inline(always)]
    |a| { yield if a { 7 } else { 13 } }
}
