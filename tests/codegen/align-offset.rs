//@ compile-flags: -Copt-level=3

#![crate_type = "lib"]

// CHECK-LABEL: @align8
#[no_mangle]
pub fn align8(p: *const u8) -> bool {
    // CHECK: ret i1 true
    p.align_offset(8) < 8
}

#[repr(align(4))]
pub struct Align4([u8; 4]);

// CHECK-LABEL: @align_to4
#[no_mangle]
pub fn align_to4(x: &[u8]) -> bool {
    // CHECK: ret i1 true
    let (prefix, _middle, suffix) = unsafe { x.align_to::<Align4>() };
    prefix.len() < 4 && suffix.len() < 4
}

// CHECK-LABEL: @align_offset_byte_ptr(ptr{{.+}}%ptr)
#[no_mangle]
pub fn align_offset_byte_ptr(ptr: *const u8) -> usize {
    // CHECK: %[[ADDR:.+]] = ptrtoint ptr %ptr to [[USIZE:i[0-9]+]]
    // CHECK: %[[UP:.+]] = add [[USIZE]] %[[ADDR]], 31
    // CHECK: %[[ALIGNED:.+]] = and [[USIZE]] %[[UP]], -32
    // CHECK: %[[OFFSET:.+]] = sub [[USIZE]] %[[ALIGNED]], %[[ADDR]]

    // Since we're offsetting a byte pointer, there's no further fixups
    // CHECK-NOT: shr
    // CHECK-NOT: div
    // CHECK-NOT: select

    // CHECK: ret [[USIZE]] %[[OFFSET]]
    ptr.align_offset(32)
}

// CHECK-LABEL: @align_offset_word_slice(ptr{{.+}}align 4{{.+}}%slice.0
#[no_mangle]
pub fn align_offset_word_slice(slice: &[Align4]) -> usize {
    // CHECK: %[[ADDR:.+]] = ptrtoint ptr %slice.0 to [[USIZE]]
    // CHECK: %[[UP:.+]] = add [[USIZE]] %[[ADDR]], 31
    // CHECK: %[[ALIGNED:.+]] = and [[USIZE]] %[[UP]], -32
    // CHECK: %[[BOFFSET:.+]] = sub [[USIZE]] %[[ALIGNED]], %[[ADDR]]
    // CHECK: %[[OFFSET:.+]] = lshr exact [[USIZE]] %[[BOFFSET]], 2

    // Slices are known to be aligned, so we don't need the "maybe -1" path
    // CHECK-NOT: select

    // CHECK: ret [[USIZE]] %[[OFFSET]]
    slice.as_ptr().align_offset(32)
}

// CHECK-LABEL: @align_offset_word_ptr(ptr{{.+}}%ptr
#[no_mangle]
pub fn align_offset_word_ptr(ptr: *const Align4) -> usize {
    // CHECK: %[[ADDR:.+]] = ptrtoint ptr %ptr to [[USIZE]]
    // CHECK: %[[UP:.+]] = add [[USIZE]] %[[ADDR]], 31
    // CHECK: %[[ALIGNED:.+]] = and [[USIZE]] %[[UP]], -32
    // CHECK: %[[BOFFSET:.+]] = sub [[USIZE]] %[[ALIGNED]], %[[ADDR]]

    // While we can always get a *byte* offset that will work, if the original
    // pointer is unaligned it might be impossible to return an *element* offset
    // that will make it aligned. We want it to be a `select`, not a `br`, so
    // that the assembly will be branchless.
    // CHECK: %[[LOW:.+]] = and [[USIZE]] %[[ADDR]], 3
    // CHECK: %[[ORIGINAL_ALIGNED:.+]] = icmp eq [[USIZE]] %[[LOW]], 0
    // CHECK: %[[OFFSET:.+]] = lshr exact [[USIZE]] %[[BOFFSET]], 2
    // CHECK: %[[R:.+]] = select i1 %[[ORIGINAL_ALIGNED]], [[USIZE]] %[[OFFSET]], [[USIZE]] -1

    // CHECK: ret [[USIZE]] %[[R]]
    ptr.align_offset(32)
}
