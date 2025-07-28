//@ compile-flags: -O -Zmir-opt-level=2
// EMIT_MIR_FOR_EACH_PANIC_STRATEGY

#![crate_type = "lib"]
#![feature(step_trait)]

// EMIT_MIR drop_boxed_slice.generic_in_place.PreCodegen.after.mir
pub unsafe fn generic_in_place<T: Copy>(ptr: *mut Box<[T]>) {
    // CHECK-LABEL: fn generic_in_place(_1: *mut Box<[T]>)
    // CHECK: (inlined <Box<[T]> as Drop>::drop)
    // CHECK: _7 = copy _6 as std::ptr::Alignment (Transmute);
    // CHECK: _9 = copy (_7.0: std::ptr::alignment::AlignmentEnum);
    // CHECK: _10 = discriminant(_9);
    // CHECK: _11 = move _10 as usize (IntToInt);
    // CHECK: = alloc::alloc::__rust_dealloc(move _8, move _5, move _11) ->
    std::ptr::drop_in_place(ptr)
}
