// check that we don't emit unneeded `resume` cleanup blocks for every
// destructor.

// CHECK-NOT: Unwind

#![feature(test)]
#![crate_type="rlib"]

extern crate test;

struct Foo {}

impl Drop for Foo {
    fn drop(&mut self) {
        test::black_box(());
    }
}

#[no_mangle]
pub fn foo() {
    let _foo = Foo {};
}
