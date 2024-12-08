//@compile-flags: -Zmiri-tree-borrows

// copy_nonoverlapping works regardless of the order in which we construct
// the arguments.
pub fn main() {
    test_to_from();
    test_from_to();
}

fn test_to_from() {
    unsafe {
        let data = &mut [0u64, 1];
        let to = data.as_mut_ptr().add(1);
        let from = data.as_ptr();
        std::ptr::copy_nonoverlapping(from, to, 1);
    }
}

// Stacked Borrows would not have liked this one because the `as_mut_ptr` reborrow
// invalidates the earlier pointer obtained from `as_ptr`, but Tree Borrows is fine
// with it.
fn test_from_to() {
    unsafe {
        let data = &mut [0u64, 1];
        let from = data.as_ptr();
        let to = data.as_mut_ptr().add(1);
        std::ptr::copy_nonoverlapping(from, to, 1);
    }
}
