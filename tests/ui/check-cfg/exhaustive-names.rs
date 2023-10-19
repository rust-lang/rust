// Check warning for unexpected cfg
//
// check-pass
// revisions: empty_names exhaustive_names
// [empty_names]compile-flags: --check-cfg=names() -Z unstable-options
// [exhaustive_names]compile-flags: --check-cfg=cfg() -Z unstable-options

#[cfg(unknown_key = "value")]
//~^ WARNING unexpected `cfg` condition name
pub fn f() {}

fn main() {}
