// Blank line after this one because it must come before `instance_user_{a,b}_rlib`.
extern crate instance_user_dylib;

extern crate instance_user_a_rlib;
extern crate instance_user_b_rlib;

use std::cell::Cell;

fn main() {
    instance_user_a_rlib::foo();
    instance_user_b_rlib::foo();
    instance_user_dylib::foo();

    let a: Cell<i32> = Cell::new(1);
    a.set(123);
}
