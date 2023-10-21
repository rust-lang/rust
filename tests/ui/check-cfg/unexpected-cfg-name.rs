// Check warning for unexpected configuration name
//
// check-pass
// revisions: names exhaustive
// compile-flags: -Z unstable-options
// [names]compile-flags: --check-cfg=names() --check-cfg=names(names,exhaustive)
// [exhaustive]compile-flags: --check-cfg=cfg() --check-cfg=cfg(names,exhaustive)

#[cfg(widnows)]
//~^ WARNING unexpected `cfg` condition name
pub fn f() {}

#[cfg(windows)]
pub fn g() {}

pub fn main() {}
