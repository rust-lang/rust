//@ test-mir-pass: RemoveUnneededDrops
//@ needs-unwind

// EMIT_MIR remove_unneeded_drop_in_place.slice_in_place.RemoveUnneededDrops.diff
unsafe fn slice_in_place(ptr: *mut [char]) {
    std::ptr::drop_in_place(ptr)
}

fn main() {
    // CHECK-LABEL: fn main(
    let mut a = ['o', 'k'];
    unsafe { slice_in_place(&raw mut a) };
}
