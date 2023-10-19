// Check warning for unexpected cfg value
//
// check-pass
// revisions: empty_values empty_cfg without_names
// [empty_values]compile-flags: --check-cfg=values() -Z unstable-options
// [empty_cfg]compile-flags: --check-cfg=cfg() -Z unstable-options
// [without_names]compile-flags: --check-cfg=cfg(any()) -Z unstable-options

#[cfg(test = "value")]
//~^ WARNING unexpected `cfg` condition value
pub fn f() {}

fn main() {}
