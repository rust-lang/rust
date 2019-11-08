// ignore-x86 FIXME: missing sysroot spans (#53081)
use std::cell::Cell;
use std::panic::catch_unwind;
fn main() {
    let mut x = Cell::new(22);
    catch_unwind(|| { x.set(23); });
    //~^ ERROR the type `std::cell::UnsafeCell<i32>` may contain interior mutability and a
}
