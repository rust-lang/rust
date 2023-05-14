// compile-flags: -O -Zmir-opt-level=2 -Cdebuginfo=0
// ignore-debug
// edition: 2021

#![crate_type = "lib"]

// EMIT_MIR transient_enums.checked_as_u32.PreCodegen.after.mir
pub fn checked_as_u32(t: u64) -> Option<u32> {
    t.try_into().ok()
}
