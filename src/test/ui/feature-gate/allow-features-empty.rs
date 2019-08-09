// compile-flags: -Z allow_features=
// Note: This test uses rustc internal flags because they will never stabilize.

#![feature(rustc_diagnostic_macros)] //~ ERROR

#![feature(rustc_const_unstable)] //~ ERROR

#![feature(lang_items)] //~ ERROR

#![feature(unknown_stdlib_feature)] //~ ERROR

fn main() {}
