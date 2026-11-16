//@ only-x86_64
#[target_feature(enable = "movdir64b")]
//~^ ERROR: currently unstable
unsafe fn foo() {}

fn main() {}
