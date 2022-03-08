// Check warning for invalid configuration value
//
// edition:2018
// check-pass
// compile-flags: --check-cfg=values(feature,"serde","full") --cfg=feature="rand" -Z unstable-options

#[cfg(feature = "sedre")]
//~^ WARNING unexpected `cfg` condition value
pub fn f() {}

#[cfg(feature = "serde")]
pub fn g() {}

#[cfg(feature = "rand")]
pub fn h() {}

pub fn main() {}
