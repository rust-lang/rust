// Check warning for unexpected cfg
//
// check-pass
// compile-flags: --check-cfg=names() -Z unstable-options

#[cfg(unknown_key = "value")]
//~^ WARNING unexpected `cfg` condition name
pub fn f() {}

fn main() {}
