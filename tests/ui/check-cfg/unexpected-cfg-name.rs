// Check warning for unexpected configuration name
//
// check-pass
// revisions: names exhaustive
// compile-flags: --check-cfg=cfg(names,exhaustive)
// [names]compile-flags: --check-cfg=names() -Z unstable-options
// [exhaustive]compile-flags: --check-cfg=cfg() -Z unstable-options

#[cfg(widnows)]
//~^ WARNING unexpected `cfg` condition name
pub fn f() {}

#[cfg(windows)]
pub fn g() {}

pub fn main() {}
