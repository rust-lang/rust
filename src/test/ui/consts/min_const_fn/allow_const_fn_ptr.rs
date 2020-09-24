#![feature(rustc_attrs, staged_api, allow_internal_unstable)]
#![feature(const_fn_fn_ptr_basics)]

#[stable(feature = "rust1", since = "1.0.0")]
#[rustc_const_stable(since="1.0.0", feature = "mep")]
const fn error(_: fn()) {}
//~^ ERROR const-stable function cannot use `#[feature(const_fn_fn_ptr_basics)]`

#[stable(feature = "rust1", since = "1.0.0")]
#[rustc_const_stable(since="1.0.0", feature = "mep")]
#[allow_internal_unstable(const_fn_fn_ptr_basics)]
const fn compiles(_: fn()) {}

fn main() {}
