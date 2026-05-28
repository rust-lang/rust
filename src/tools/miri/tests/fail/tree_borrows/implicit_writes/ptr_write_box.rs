// same test as `ptr_write.rs` but with `Box`
//@compile-flags: -Zmiri-tree-borrows -Zmiri-tree-borrows-implicit-writes

fn main() {
    let mut x: Box<u8> = Box::new(0u8);
    let ptr = &raw mut *x;
    let res = dereference(x, ptr);
    assert_eq!(*res, 0);
}

fn dereference<T>(x: T, y: *mut u8) -> T {
    // miri inserts an implicit write here
    let _ = unsafe { *y }; //~ ERROR: read access
    // x = 42; // we'd like to add this write, so there must already be UB without it because there sure is with it.
    x
}
