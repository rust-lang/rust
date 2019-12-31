// Test internal const fn feature gate.

#![feature(const_fn)]

#[rustc_const_unstable(feature="fzzzzzt")] //~ stability attributes may not be used outside
pub const fn bazinga() {}

fn main() {
}
