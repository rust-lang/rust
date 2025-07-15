//@ compile-flags: -O -Zmir-opt-level=2
// EMIT_MIR_FOR_EACH_PANIC_STRATEGY
// EMIT_MIR_FOR_EACH_BIT_WIDTH

#![crate_type = "lib"]

// EMIT_MIR drop_boxed_slice.generic_in_place.PreCodegen.after.mir
pub unsafe fn generic_in_place<T: Copy>(ptr: *mut Box<[T]>) {
    // CHECK-LABEL: fn generic_in_place(_1: *mut Box<[T]>)
    // CHECK: (inlined <Box<[T]> as Drop>::drop)
    // CHECK: [[SIZE:_.+]] = std::intrinsics::size_of_val::<[T]>
    // CHECK: [[ALIGN:_.+]] = AlignOf(T);
    // CHECK: [[B:_.+]] = copy [[ALIGN]] as std::ptr::Alignment (Transmute);
    // CHECK: [[C:_.+]] = move ([[B]].0: std::ptr::alignment::AlignmentEnum);
    // CHECK: [[D:_.+]] = discriminant([[C]]);
    // CHECK: = alloc::alloc::__rust_dealloc({{.+}}, move [[SIZE]], move [[D]]) ->
    std::ptr::drop_in_place(ptr)
}
