// Test internal const fn feature gate.

#![feature(staged_api)]
#![feature(const_fn)]
//#![feature(rustc_const_unstable)]

#[stable(feature="zing", since="1.0.0")]
#[rustc_const_unstable(feature="fzzzzzt")] //~ERROR internal feature
pub const fn bazinga() {}

fn main() {
}
