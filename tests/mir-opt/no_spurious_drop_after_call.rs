// EMIT_MIR_FOR_EACH_PANIC_STRATEGY

// Test that after the call to `std::mem::drop` we do not generate a
// MIR drop of the argument. (We used to have a `DROP(_2)` in the code
// below, as part of bb3.)

// EMIT_MIR no_spurious_drop_after_call.main.ElaborateDrops.before.mir
fn main() {
    std::mem::drop("".to_string());
}
