//@revisions: stack tree
//@[tree]compile-flags: -Zmiri-tree-borrows

use std::alloc;
use std::cell::Cell;

type T = (Cell<i32>, i32);

// Deallocating `x` is UB because not all bytes are in an `UnsafeCell`.
fn foo(x: &T) {
    let layout = alloc::Layout::new::<T>();
    unsafe { alloc::dealloc(x as *const _ as *mut T as *mut u8, layout) }; //~ERROR: dealloc
}

fn main() {
    let b: Box<T> = Box::new((Cell::new(0), 0));
    foo(unsafe { std::mem::transmute(Box::into_raw(b)) });
}
