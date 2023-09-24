// revisions: mir thir
// [thir]compile-flags: -Z thir-unsafeck
// only-x86_64

#[target_feature(enable = "sse2")]
fn foo() {}

fn main() {
    let foo: fn() = foo; //~ ERROR mismatched types
}
