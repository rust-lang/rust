#![feature(rustc_attrs, staged_api)]

#[stable(feature = "rust1", since = "1.0.0")]
const fn error(_: fn()) {} //~ ERROR function pointers in const fn are unstable

#[stable(feature = "rust1", since = "1.0.0")]
#[rustc_allow_const_fn_ptr]
const fn compiles(_: fn()) {}

fn main() {}
