// Tests that UB is detected for reading from a mutable reference by another pointer (as there could be a write on that mutable reference now)
//@compile-flags: -Zmiri-tree-borrows -Zmiri-tree-borrows-implicit-writes

fn main() {
    let mut x = 0u8;
    let ptr = &raw mut x;
    let res = dereference(&mut x, ptr);
    assert_eq!(*res, 0);
}

fn dereference<T>(x: T, y: *mut u8) -> T {
    // miri inserts an implicit write here
    let _ = unsafe { *y }; //~ ERROR: read access
    // x = 42; // we'd like to add this write, so there must already be UB without it because there sure is with it.
    x
}
