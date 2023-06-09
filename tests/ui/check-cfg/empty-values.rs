// Check warning for unexpected cfg value
//
// check-pass
// compile-flags: --check-cfg=values() -Z unstable-options

#[cfg(test = "value")]
//~^ WARNING unexpected `cfg` condition value
pub fn f() {}

fn main() {}
