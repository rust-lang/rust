// Check warning for invalid configuration name
//
// edition:2018
// check-pass
// compile-flags: --check-cfg=names() -Z unstable-options

#[cfg(widnows)]
//~^ WARNING unexpected `cfg` condition name
pub fn f() {}

#[cfg(windows)]
pub fn g() {}

pub fn main() {}
