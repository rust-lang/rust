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
    // TODO: Ideally, this would result in UB since the second element
    // in `pair` is Frozen.  We would need some way to express a
    // "shared reference with permission to access surrounding
    // interior mutable data".
    foo(&pair.0);
}
