// should find the bug even without retagging
//@compile-flags: -Zmiri-disable-stacked-borrows

struct SliceWithHead(#[allow(dead_code)] u8, #[allow(dead_code)] [u8]);

fn main() {
    let buf = [0u32; 1];
    // We craft a wide pointer `*const SliceWithHead` such that the unsized tail is only partially allocated.
    // That should lead to UB, as the reference is not fully dereferenceable.
    let ptr: *const SliceWithHead = unsafe { std::mem::transmute((&buf, 4usize)) };
    // Re-borrow that. This should be UB.
    let _ptr = unsafe { &*ptr }; //~ ERROR: encountered a dangling reference (going beyond the bounds of its allocation)
}
