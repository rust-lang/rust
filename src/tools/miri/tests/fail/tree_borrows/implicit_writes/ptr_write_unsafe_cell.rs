// same test as `ptr_write.rs` but with `UnsafeCell`
//@compile-flags: -Zmiri-tree-borrows -Zmiri-tree-borrows-implicit-writes

use std::cell::UnsafeCell;

fn main() {
    let mut x: UnsafeCell<u8> = 0u8.into();
    let ptr = x.get();
    let res = dereference(&mut x, ptr);
    assert_eq!(*res.get_mut(), 0);
}

fn dereference<T>(x: T, y: *mut u8) -> T {
    // miri inserts an implicit write here
    let _ = unsafe { *y }; //~ ERROR: read access
    // x = 42; // we'd like to add this write, so there must already be UB without it because there sure is with it.
    x
}
