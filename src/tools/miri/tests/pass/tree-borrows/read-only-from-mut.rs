//@compile-flags: -Zmiri-tree-borrows

// Tree Borrows has no issue with several mutable references existing
// at the same time, as long as they are used only immutably.
// I.e. multiple Reserved can coexist.
pub fn main() {
    unsafe {
        let base = &mut 42u64;
        let r1 = &mut *(base as *mut u64);
        let r2 = &mut *(base as *mut u64);
        let _l = *r1;
        let _l = *r2;
    }
}
