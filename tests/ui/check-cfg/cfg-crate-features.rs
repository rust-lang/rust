// https://github.com/rust-lang/rust/issues/143977
// Check that features are available when an attribute is applied to a crate

#![cfg(version("1.0"))]
//~^ ERROR `cfg(version)` is experimental and subject to change

// Using invalid value `does_not_exist`,
// so we don't accidentally configure out the crate for any certain OS
#![cfg(not(target(os = "does_not_exist")))]
//~^ ERROR compact `cfg(target(..))` is experimental and subject to change
//~| WARN unexpected `cfg` condition value: `does_not_exist`

fn main() {}
