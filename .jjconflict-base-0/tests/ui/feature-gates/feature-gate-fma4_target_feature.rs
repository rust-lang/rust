//@ only-x86_64
#[target_feature(enable = "fma4")]
//~^ ERROR: currently unstable
unsafe fn foo() {}

fn main() {}
