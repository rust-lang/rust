//! Reborrowing a `&mut !Unpin` must still act like a (fake) read.
use std::marker::PhantomPinned;

struct NotUnpin(i32, PhantomPinned);

fn main() {
    unsafe {
        let mut x = NotUnpin(0, PhantomPinned);
        // Mutable borrow of `Unpin` field (with lifetime laundering)
        let fieldref = &mut *(&mut x.0 as *mut i32);
        // Mutable reborrow of the entire `x`, which is `!Unpin` but should
        // still count as a read since we would add `dereferenceable`.
        let _xref = &mut x;
        // That read should have invalidated `fieldref`.
        *fieldref = 0; //~ ERROR: /write access .* tag does not exist in the borrow stack/
    }
}
