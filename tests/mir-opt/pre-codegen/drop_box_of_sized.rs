//@ compile-flags: -O -Zmir-opt-level=2
// EMIT_MIR_FOR_EACH_PANIC_STRATEGY

#![crate_type = "lib"]

// EMIT_MIR drop_box_of_sized.drop_generic.PreCodegen.after.mir
pub unsafe fn drop_generic<T: Copy>(x: *mut Box<T>) {
    // CHECK-LABEL: fn drop_generic
    // CHECK: [[ALIGNMENT:_.+]] = const <T as std::mem::SizedTypeProperties>::ALIGN as std::ptr::Alignment (Transmute)
    // CHECK: alloc::alloc::__rust_dealloc({{.+}}, const <T as std::mem::SizedTypeProperties>::SIZE, move [[ALIGNMENT]])
    std::ptr::drop_in_place(x)
}

// EMIT_MIR drop_box_of_sized.drop_bytes.PreCodegen.after.mir
pub unsafe fn drop_bytes(x: *mut Box<[u8; 1024]>) {
    // CHECK-LABEL: fn drop_bytes
    // CHECK: alloc::alloc::__rust_dealloc({{.+}}, const 1024_usize, {{.+}}Align1Shl0 {{.+}})
    std::ptr::drop_in_place(x)
}
