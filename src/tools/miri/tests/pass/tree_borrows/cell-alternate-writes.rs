// We disable the GC for this test because it would change what is printed.
//@compile-flags: -Zmiri-tree-borrows -Zmiri-provenance-gc=0
#[path = "../../utils/mod.rs"]
#[macro_use]
mod utils;

use std::cell::UnsafeCell;

// UnsafeCells use the parent tag, so it is possible to use them with
// few restrictions when only among themselves.
fn main() {
    unsafe {
        let data = &mut UnsafeCell::new(0u8);
        name!(data.get(), "data");
        let x = &*data;
        name!(x.get(), "x");
        let y = &*data;
        name!(y.get(), "y");
        let alloc_id = alloc_id!(data.get());
        print_state!(alloc_id);
        // y and x tolerate alternating Writes
        *y.get() = 1;
        *x.get() = 2;
        *y.get() = 3;
        *x.get() = 4;
        print_state!(alloc_id);
    }
}
