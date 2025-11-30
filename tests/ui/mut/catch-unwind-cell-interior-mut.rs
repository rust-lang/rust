//@ compile-flags: -Zwrite-long-types-to-disk=yes
use std::cell::Cell;
use std::panic::catch_unwind;
fn main() {
    let mut x = Cell::new(22);
    catch_unwind(|| { x.set(23); });
    //~^ ERROR the type `UnsafeCell<i32>` may contain interior mutability and a
}
