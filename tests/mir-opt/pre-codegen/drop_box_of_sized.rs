//@ compile-flags: -O -Zmir-opt-level=2
// EMIT_MIR_FOR_EACH_PANIC_STRATEGY

#![crate_type = "lib"]

// EMIT_MIR drop_box_of_sized.drop_generic.PreCodegen.after.mir
pub unsafe fn drop_generic<T: Copy>(x: *mut Box<T>) {
    // CHECK-LABEL: fn drop_generic
    // CHECK: = std::intrinsics::size_of_val::<T>
    // CHECK: = std::intrinsics::align_of_val::<T>
    // CHECK: alloc::alloc::__rust_dealloc
    std::ptr::drop_in_place(x)
}

// EMIT_MIR drop_box_of_sized.drop_bytes.PreCodegen.after.mir
pub unsafe fn drop_bytes(x: *mut Box<[u8; 1024]>) {
    // CHECK-LABEL: fn drop_bytes
    // CHECK: = std::intrinsics::size_of_val::<[u8; 1024]>
    // CHECK: = std::intrinsics::align_of_val::<[u8; 1024]>
    // CHECK: alloc::alloc::__rust_dealloc
    std::ptr::drop_in_place(x)
}
