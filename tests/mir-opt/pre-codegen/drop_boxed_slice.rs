//@ compile-flags: -O -Zmir-opt-level=2
// EMIT_MIR_FOR_EACH_PANIC_STRATEGY

#![crate_type = "lib"]
#![feature(step_trait)]

// EMIT_MIR drop_boxed_slice.generic_in_place.PreCodegen.after.mir
pub unsafe fn generic_in_place<T: Copy>(ptr: *mut Box<[T]>) {
    // CHECK-LABEL: fn generic_in_place(_1: *mut Box<[T]>)
    // CHECK: (inlined <Box<[T]> as Drop>::drop)
    // CHECK: [[SIZE:_.+]] = std::intrinsics::size_of_val::<[T]>
    // CHECK: [[ALIGN:_.+]] = std::intrinsics::align_of_val::<[T]>
    // CHECK: [[ALIGNMENT:_.+]] = copy [[ALIGN]] as std::ptr::Alignment (Transmute);
    // CHECK-NOT: discriminant
    // CHECK-NOT: IntToInt
    // CHECK: = alloc::alloc::__rust_dealloc({{.+}}, move [[SIZE]], move [[ALIGNMENT]]) ->
    std::ptr::drop_in_place(ptr)
}
