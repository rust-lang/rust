// compile-flags: -Zmiri-disable-stacked-borrows
// normalize-stderr-test: "offset -[0-9]+" -> "offset -XX"
#![feature(strict_provenance)]

use std::ptr;

// Make sure that with legacy provenance, the allocation id of
// a casted pointer is determined at cast-time
fn main() {
    let x: i32 = 0;
    let y: i32 = 1;

    let x_ptr = &x as *const i32;
    let y_ptr = &y as *const i32;

    let x_usize = x_ptr.expose_addr();
    let y_usize = y_ptr.expose_addr();

    let ptr = ptr::from_exposed_addr::<i32>(y_usize);
    let ptr = ptr.with_addr(x_usize);
    assert_eq!(unsafe { *ptr }, 0); //~ ERROR is out-of-bounds
}
