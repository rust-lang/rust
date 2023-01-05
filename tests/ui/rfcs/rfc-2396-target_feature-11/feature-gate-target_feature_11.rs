// only-x86_64

#[target_feature(enable = "sse2")] //~ ERROR can only be applied to `unsafe` functions
fn foo() {}

fn main() {}
