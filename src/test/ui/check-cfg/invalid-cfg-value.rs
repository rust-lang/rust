// Check warning for invalid configuration value
//
// edition:2018
// check-pass
// compile-flags: --check-cfg=values(feature,"serde") --cfg=feature="rand" -Z unstable-options

#[cfg(feature = "sedre")]
//~^ WARNING unknown condition value used
pub fn f() {}

#[cfg(feature = "serde")]
pub fn g() {}

#[cfg(feature = "rand")]
pub fn h() {}

pub fn main() {}
