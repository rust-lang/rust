//@compile-flags: -Zmiri-strict-provenance

// Taken from <https://github.com/rust-lang/unsafe-code-guidelines/issues/194#issuecomment-520934222>.

use std::cell::{Cell, UnsafeCell};

fn helper(val: Box<Cell<u8>>, ptr: *const UnsafeCell<u8>) -> u8 {
    val.set(10);
    unsafe { (*ptr).get().write(20) }; //~ ERROR: protected
    val.get()
}

fn main() {
    let val: Box<Cell<u8>> = Box::new(Cell::new(25));
    let ptr: *const Cell<u8> = &*val;
    let res = helper(val, ptr.cast());
    assert_eq!(res, 20);
}
