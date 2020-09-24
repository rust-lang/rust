#![feature(staged_api)]

#[stable(feature = "rust1", since = "1.0.0")]
const fn error(_: fn()) {}
//~^ ERROR `rustc_const_stable` or `rustc_const_unstable`
//~| ERROR `rustc_const_stable` or `rustc_const_unstable`
//~| ERROR function pointers

#[stable(feature = "rust1", since = "1.0.0")]
#[rustc_const_stable(since="1.0.0", feature = "mep")]
const fn compiles(_: fn()) {}
//~^ ERROR function pointers

fn main() {}
