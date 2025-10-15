//@compile-flags: -Zmiri-tree-borrows

use std::cell::Cell;

fn foo(x: &Cell<i32>) {
    unsafe {
        let ptr = x as *const Cell<i32> as *mut Cell<i32> as *mut i32;
        ptr.offset(1).write(0);
    }
}

fn main() {
    let arr = [Cell::new(1), Cell::new(1)];
    foo(&arr[0]);

    let pair = (Cell::new(1), 1);
    foo(&pair.0);

    // As long as the "inside" part is `!Freeze`, the permission to mutate the "outside" is preserved.
    let pair = (Cell::new(()), 1);
    let x = &pair.0;
    let ptr = (&raw const *x).cast::<i32>().cast_mut();
    unsafe { ptr.write(0) };
}
