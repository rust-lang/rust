//@ only-x86_64
#[target_feature(enable = "avx10.1")]
//~^ ERROR: currently unstable
unsafe fn foo() {}

fn main() {}
