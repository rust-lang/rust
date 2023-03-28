// compile-flags: -Zpanic-in-drop=abort -Copt-level=1 --crate-type lib
// check-pass
#![warn(panic_in_drop)]

pub struct A;

impl Drop for A {
    fn drop(&mut self) {
        (|| bar())();
    }
}

fn bar() {
    todo!();
}
