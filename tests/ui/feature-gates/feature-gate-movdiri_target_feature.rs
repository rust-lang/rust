//@ only-x86_64
#[target_feature(enable = "movdiri")]
//~^ ERROR: currently unstable
unsafe fn foo() {}

fn main() {}
