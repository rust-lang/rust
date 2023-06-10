// EMIT_MIR_FOR_EACH_PANIC_STRATEGY
// compile-flags: -Zmir-opt-level=2 -Zinline-mir

#![crate_type = "lib"]

// EMIT_MIR private_helper.outer.Inline.diff
pub fn outer() -> u8 {
    helper()
}

fn helper() -> u8 {
    123
}
