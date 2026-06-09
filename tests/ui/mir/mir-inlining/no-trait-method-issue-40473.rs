//@ run-pass
//@ compile-flags:-Zmir-opt-level=3
pub trait Foo {
    fn bar(&self) -> usize { 2 }
}

impl Foo for () {
    fn bar(&self) -> usize { 3 }
}

// Test a case where MIR would inline the default trait method
// instead of bailing out. Issue #40473.
fn main() {
    let result = ().bar();
    assert_eq!(result, 3);
}
