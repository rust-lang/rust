// Check for unexpected configuration value in the code.
//
//@ check-pass
//@ compile-flags: --cfg=feature="rand" -Z unstable-options
//@ compile-flags: --check-cfg=cfg(feature,values("serde","full"))

#[cfg(feature = "sedre")]
//~^ WARNING unexpected `cfg` condition value
pub fn f() {}

#[cfg(feature = "serde")]
pub fn g() {}

#[cfg(feature = "rand")]
//~^ WARNING unexpected `cfg` condition value
pub fn h() {}

pub fn main() {}
