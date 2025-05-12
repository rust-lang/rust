//@ only-x86_64
#[target_feature(enable = "amx-tile")]
//~^ ERROR: currently unstable
unsafe fn foo() {}

fn main() {}
