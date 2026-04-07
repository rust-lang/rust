// Tests that the permissions are as expected after reborrowing.
// To be precise, this tests that at the start of a function, a mutable reference is Unique.
//@compile-flags: -Zmiri-tree-borrows

unsafe extern "Rust" {
    safe fn miri_get_alloc_id(ptr: *const u8) -> u64;
    safe fn miri_print_borrow_state(alloc_id: u64, show_unnamed: bool);
}

fn bar(x: &mut u8) {
    miri_print_borrow_state(miri_get_alloc_id(x), true);
}

fn main() {
    let mut x = 0u8;
    bar(&mut x);
}
