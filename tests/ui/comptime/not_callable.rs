#![feature(rustc_attrs)]

//TODO: should fail
//@check-pass

#[rustc_comptime]
fn foo() {}

fn main() {
    // Ok
    const { foo() };
    // TODO: Not ok
    foo();
}

const fn bar() {
    // TODO: Not ok
    foo();
}

#[rustc_comptime]
fn baz() {
    // Should be allowed
    foo();
}
