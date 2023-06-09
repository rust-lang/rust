use std::cell::Cell;

pub fn foo() {
    let a: Cell<i32> = Cell::new(1);
    a.set(123);
}
