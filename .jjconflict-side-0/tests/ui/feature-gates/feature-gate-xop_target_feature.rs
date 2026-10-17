//@ only-x86_64
#[target_feature(enable = "xop")]
//~^ ERROR: currently unstable
unsafe fn foo() {}

fn main() {}
