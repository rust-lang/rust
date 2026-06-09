// This test ensures proper diagnostics emission during HIR ty lowering
// See https://github.com/rust-lang/rust/issues/153254

#![feature(min_generic_const_args)]
#![expect(incomplete_features)]

type const T0: _ = ();
//~^ ERROR: the placeholder `_` is not allowed within types on item signatures for constants [E0121]

type const T1 = [0];
//~^ ERROR: missing type for `const` item

fn main() {}
