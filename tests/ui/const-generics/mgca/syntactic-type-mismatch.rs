// This test ensures proper diagnostics emission during HIR ty lowering
// See https://github.com/rust-lang/rust/issues/153254

#![feature(min_generic_const_args)]
#![expect(incomplete_features)]

const T0: _ = core::direct_const_arg!(());
//~^ ERROR: the placeholder `_` is not allowed within types on item signatures for constants [E0121]

const T1 = core::direct_const_arg!([0]);
//~^ ERROR: missing type for `const` item

fn main() {}
