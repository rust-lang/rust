extern crate instance_provider_a as upstream;
use std::cell::Cell;

pub fn foo() {
    upstream::foo();

    let b: Cell<i32> = Cell::new(1);
    b.set(123);
}
