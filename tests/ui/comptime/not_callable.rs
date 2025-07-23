#![feature(rustc_attrs)]

// TODO: this test should fail
//@check-pass

#[rustc_comptime]
fn foo() {}

fn main() {
    // Ok
    const { foo() };
    // Not ok
    foo();
}
