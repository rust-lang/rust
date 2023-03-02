// revisions: mir thir
// [thir]compile-flags: -Z thir-unsafeck
// only-x86_64

#![feature(target_feature_11)]

#[target_feature(enable = "sse2")]
fn foo() {}

fn main() {
    let foo: fn() = foo; //~ ERROR mismatched types
}
