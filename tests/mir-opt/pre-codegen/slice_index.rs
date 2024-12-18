//@ compile-flags: -O -C debuginfo=0 -Zmir-opt-level=2 -Z ub-checks=yes
// EMIT_MIR_FOR_EACH_PANIC_STRATEGY

#![crate_type = "lib"]
#![feature(slice_ptr_get)]

use std::ops::Range;

// EMIT_MIR slice_index.slice_index_usize.PreCodegen.after.mir
pub fn slice_index_usize(slice: &[u32], index: usize) -> u32 {
    // CHECK-LABEL: slice_index_usize
    // CHECK: [[LEN:_[0-9]+]] = PtrMetadata(copy _1)
    // CHECK: Lt(copy _2, copy [[LEN]])
    // CHECK-NOT: precondition_check
    // CHECK: _0 = copy (*_1)[_2];
    slice[index]
}

// EMIT_MIR slice_index.slice_get_mut_usize.PreCodegen.after.mir
pub fn slice_get_mut_usize(slice: &mut [u32], index: usize) -> Option<&mut u32> {
    // CHECK-LABEL: slice_get_mut_usize
    // CHECK: [[LEN:_[0-9]+]] = PtrMetadata(copy _1)
    // CHECK: Lt(copy _2, move [[LEN]])
    // CHECK-NOT: precondition_check
    slice.get_mut(index)
}

// EMIT_MIR slice_index.slice_index_range.PreCodegen.after.mir
pub fn slice_index_range(slice: &[u32], index: Range<usize>) -> &[u32] {
    // CHECK-LABEL: slice_index_range
    &slice[index]
}

// EMIT_MIR slice_index.slice_get_unchecked_mut_range.PreCodegen.after.mir
pub unsafe fn slice_get_unchecked_mut_range(slice: &mut [u32], index: Range<usize>) -> &mut [u32] {
    // CHECK-LABEL: slice_get_unchecked_mut_range
    // CHECK: [[START:_[0-9]+]] = move (_2.0: usize);
    // CHECK: [[END:_[0-9]+]] = move (_2.1: usize);
    // CHECK: precondition_check
    // CHECK: [[LEN:_[0-9]+]] = SubUnchecked(copy [[END]], copy [[START]]);
    // CHECK: [[PTR:_[0-9]+]] = Offset(copy {{_[0-9]+}}, copy [[START]]);
    // CHECK: [[SLICE:_[0-9]+]] = *mut [u32] from (copy [[PTR]], copy [[LEN]])
    // CHECK: _0 = &mut (*[[SLICE]]);
    slice.get_unchecked_mut(index)
}

// EMIT_MIR slice_index.slice_ptr_get_unchecked_range.PreCodegen.after.mir
pub unsafe fn slice_ptr_get_unchecked_range(
    slice: *const [u32],
    index: Range<usize>,
) -> *const [u32] {
    // CHECK-LABEL: slice_ptr_get_unchecked_range
    // CHECK: [[START:_[0-9]+]] = move (_2.0: usize);
    // CHECK: [[END:_[0-9]+]] = move (_2.1: usize);
    // CHECK: precondition_check
    // CHECK: [[LEN:_[0-9]+]] = SubUnchecked(copy [[END]], copy [[START]]);
    // CHECK: [[PTR:_[0-9]+]] = Offset(copy {{_[0-9]+}}, copy [[START]]);
    // CHECK: _0 = *const [u32] from (copy [[PTR]], copy [[LEN]])
    slice.get_unchecked(index)
}
