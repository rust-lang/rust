// skip-filecheck
//@ compile-flags: -O -C debuginfo=0 -Zmir-opt-level=2 -Zinline-mir
//@ ignore-std-debug-assertions (precondition checks are under cfg(debug_assertions))
// EMIT_MIR_FOR_EACH_PANIC_STRATEGY

#![crate_type = "lib"]

// EMIT_MIR ptr_offset.demo_byte_add_thin.PreCodegen.after.mir
pub unsafe fn demo_byte_add_thin(p: *const u32, n: usize) -> *const u32 {
    p.byte_add(n)
}

// EMIT_MIR ptr_offset.demo_byte_add_fat.PreCodegen.after.mir
pub unsafe fn demo_byte_add_fat(p: *const [u32], n: usize) -> *const [u32] {
    p.byte_add(n)
}
