//@ test-mir-pass: RemoveUnneededDrops
//@ needs-unwind
//@ compile-flags: -Z mir-opt-level=1

// EMIT_MIR remove_unneeded_drop_in_place.slice_in_place.RemoveUnneededDrops.diff
unsafe fn slice_in_place(ptr: *mut [char]) {
    // CHECK-LABEL: fn slice_in_place(_1: *mut [char])
    // CHECK: bb0: {
    // CHECK-NEXT: return;
    // CHECK-NEXT: }
    std::ptr::drop_in_place(ptr)
}

fn main() {
    let mut a = ['o', 'k'];
    unsafe { slice_in_place(&raw mut a) };
}
