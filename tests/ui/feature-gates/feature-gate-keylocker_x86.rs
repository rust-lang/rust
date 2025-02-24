//@ only-x86_64
#[target_feature(enable = "kl")]
//~^ ERROR: currently unstable
unsafe fn foo() {}

fn main() {}
