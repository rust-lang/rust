// should find the bug even without these
//@compile-flags: -Zmiri-disable-validation -Zmiri-disable-stacked-borrows

struct SliceWithHead(u8, [u8]);

fn main() {
    let buf = [0u32; 1];
    // We craft a wide pointer `*const SliceWithHead` such that the unsized tail is only partially allocated.
    // That should be UB, as the reference is not fully dereferencable.
    let ptr: *const SliceWithHead = unsafe { std::mem::transmute((&buf, 4usize)) };
    // Re-borrow that. This should be UB.
    let _ptr = unsafe { &*ptr }; //~ ERROR: pointer to 5 bytes starting at offset 0 is out-of-bounds
}
