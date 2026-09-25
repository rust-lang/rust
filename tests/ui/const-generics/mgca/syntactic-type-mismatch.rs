// This test ensures proper diagnostics emission during HIR ty lowering
// See https://github.com/rust-lang/rust/issues/153254

#![feature(gca_min_const_items)]
#![expect(incomplete_features)]

use std::gca;

const T0: _ = gca!(());
//~^ ERROR: the placeholder `_` is not allowed within types on item signatures for constants [E0121]

const T1 = gca!([0]);
//~^ ERROR: missing type for `const` item

fn main() {}
