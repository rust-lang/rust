//@ only-x86_64
#[target_feature(enable = "sha512")]
//~^ ERROR: currently unstable
unsafe fn foo() {}

fn main() {}
