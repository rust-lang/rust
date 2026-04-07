// Tests that UB is detected for reading from a mutable reference by another pointer (as there could be a write on that mutable reference now)
//@compile-flags: -Zmiri-tree-borrows

fn foo(_x: &mut u8, y: *mut u8) -> u8 {
    // spurious write inserted here for x
    let val = unsafe { *y }; //~ ERROR: read access
    // *x = 42; // we'd like to add this write, so there must already be UB without it because there sure is with it.
    val
}

fn main() {
    let mut x = 0u8;
    let ptr = &raw mut x;
    let res = foo(&mut x, ptr);
    assert_eq!(res, 0);
}
