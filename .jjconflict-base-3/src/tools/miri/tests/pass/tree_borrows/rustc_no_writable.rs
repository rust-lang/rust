// Tests that the `#[rustc_no_writable]` attribute disables implicit writes checking for the function it is applied to, so that `tests/fail/tree_borrows/implicit_writes/ptr_write.rs` would pass even with implicit writes enabled.
//@compile-flags: -Zmiri-tree-borrows -Zmiri-tree-borrows-implicit-writes

#![feature(rustc_attrs)]

fn main() {
    let mut x = 0u8;
    let ptr = &raw mut x;
    let res = dereference(&mut x, ptr);
    assert_eq!(*res, 0);
}

#[rustc_no_writable]
fn dereference<T>(x: T, y: *mut u8) -> T {
    // miri inserts *no* implicit write here due to the attribute.
    // Therefore we end up only reading from `x` and `y` which is fine.
    let _ = unsafe { *y };
    x
}
