// skip-filecheck
//@ compile-flags: -O -C debuginfo=0 -Zmir-opt-level=2 -Zinline-mir
//@ ignore-std-debug-assertions
// Reason: precondition checks on ptr::read/write are under cfg(debug_assertions)
// EMIT_MIR_FOR_EACH_PANIC_STRATEGY

#![crate_type = "lib"]

// EMIT_MIR mem_replace.manual_replace.PreCodegen.after.mir
pub fn manual_replace(r: &mut u32, v: u32) -> u32 {
    let temp = *r;
    *r = v;
    temp
}

// EMIT_MIR mem_replace.mem_replace.PreCodegen.after.mir
pub fn mem_replace(r: &mut u32, v: u32) -> u32 {
    std::mem::replace(r, v)
}
