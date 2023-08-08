// Test that the comments we emit in MIR opts are accurate.
//
// EMIT_MIR_FOR_EACH_PANIC_STRATEGY
// compile-flags: -Zmir-include-spans
// ignore-wasm32

#![crate_type = "lib"]

// EMIT_MIR spans.outer.PreCodegen.after.mir
pub fn outer(v: u8) -> u8 {
    inner(&v)
}

pub fn inner(x: &u8) -> u8 {
    *x
}
