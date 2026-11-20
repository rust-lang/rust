// Shows that without implicit writes, `tests/fail/tree_borrows/implicit_writes/ptr_write.rs`, `tests/fail/tree_borrows/implicit_writes/ptr_write_unsafe_cell.rs`, `tests/fail/tree_borrows/implicit_writes/ptr_write_box.rs`, `tests/fail/tree_borrows/implicit_writes/as_mut_ptr.rs` would pass.
//@compile-flags: -Zmiri-tree-borrows

use std::cell::UnsafeCell;

fn main() {
    normal();
    unsafe_cell();
    box_test();
    as_mut_ptr_test()
}

fn normal() {
    let mut x = 0u8;
    let ptr = &raw mut x;
    let res = dereference(&mut x, ptr);
    assert_eq!(*res, 0);
}

fn unsafe_cell() {
    let mut x: UnsafeCell<u8> = 0u8.into();
    let ptr = x.get();
    let res = dereference(&mut x, ptr);
    assert_eq!(*res.get_mut(), 0);
}

fn box_test() {
    let mut x: Box<u8> = Box::new(0u8);
    let ptr = &raw mut *x;
    let res = dereference(x, ptr);
    assert_eq!(*res, 0);
}

fn dereference<T>(x: T, y: *mut u8) -> T {
    // miri inserts an implicit write here
    let _ = unsafe { *y };
    // x = 42; // we'd like to add this write, so there must already be UB without it because there sure is with it.
    x
}

fn as_mut_ptr_test() {
    let mut x: [u8; 3] = [1, 2, 3];

    let ptr = std::ptr::from_mut(&mut x);
    let a = unsafe { &mut *ptr };
    let b = unsafe { &mut *ptr };

    let _c = as_mut_ptr(a);
    assert_eq!(*b, [1, 2, 3]);
}

// this should be the same as the implementation for slice, as this is known to cause errors and we want to test that behavior here
const fn as_mut_ptr(x: &mut [u8; 3]) -> *mut u8 {
    x as *mut [u8] as *mut u8
}
