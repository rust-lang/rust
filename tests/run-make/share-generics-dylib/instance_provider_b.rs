use std::cell::Cell;

pub fn foo() {
    let b: Cell<i32> = Cell::new(1);
    b.set(123);
}
