//@ only-x86_64
#[target_feature(enable = "apxf")]
//~^ ERROR: currently unstable
unsafe fn foo() {}

fn main() {}
