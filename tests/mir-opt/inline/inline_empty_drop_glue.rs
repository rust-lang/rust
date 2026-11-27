//@ test-mir-pass: Inline
//@ needs-unwind
//@ compile-flags: -Zmir-opt-level=1

// EMIT_MIR inline_empty_drop_glue.slice_in_place.Inline.diff
unsafe fn slice_in_place(ptr: *mut [char]) {
    // CHECK-LABEL: fn slice_in_place(_1: *mut [char])
    // CHECK:      bb0: {
    // CHECK-NOT:  ->
    // CHECK:        return;
    // CHECK-NEXT: }
    std::ptr::drop_in_place(ptr)
}

fn main() {
    let mut a = ['o', 'k'];
    unsafe { slice_in_place(&raw mut a) };
}
