#![feature(rustc_attrs, staged_api, rustc_allow_const_fn_unstable)]
#![feature(const_fn_fn_ptr_basics)]

#[stable(feature = "rust1", since = "1.0.0")]
#[rustc_const_stable(since="1.0.0", feature = "mep")]
const fn error(_: fn()) {}
//~^ ERROR const-stable function cannot use `#[feature(const_fn_fn_ptr_basics)]`

#[stable(feature = "rust1", since = "1.0.0")]
#[rustc_const_stable(since="1.0.0", feature = "mep")]
#[rustc_allow_const_fn_unstable(const_fn_fn_ptr_basics)]
const fn compiles(_: fn()) {}

fn main() {}
