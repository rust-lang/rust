// Check warning for invalid configuration value in the code and
// in the cli
//
// check-pass
// revisions: values cfg
// compile-flags: --cfg=feature="rand" -Z unstable-options
// [values]compile-flags: --check-cfg=values(feature,"serde","full") --check-cfg=names(values,cfg)
// [cfg]compile-flags: --check-cfg=cfg(feature,values("serde","full")) --check-cfg=cfg(values,cfg)

#[cfg(feature = "sedre")]
//~^ WARNING unexpected `cfg` condition value
pub fn f() {}

#[cfg(feature = "serde")]
pub fn g() {}

#[cfg(feature = "rand")]
//~^ WARNING unexpected `cfg` condition value
pub fn h() {}

pub fn main() {}
