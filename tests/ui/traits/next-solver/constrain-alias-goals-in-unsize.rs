//@ compile-flags: -Znext-solver
//@ check-pass

use std::mem::ManuallyDrop;

trait Foo {}

struct Guard<T> {
    value: ManuallyDrop<T>,
}

impl<T: Foo> Guard<T> {
    fn uwu(&self) {
        let x: &dyn Foo = &*self.value;
    }
}

fn main() {}
