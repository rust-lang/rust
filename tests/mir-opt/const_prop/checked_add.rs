// EMIT_MIR_FOR_EACH_PANIC_STRATEGY
//@ test-mir-pass: GVN
//@ compile-flags: -C overflow-checks=on -Zdump-mir-exclude-alloc-bytes

// EMIT_MIR checked_add.main.GVN.diff
fn main() {
    // CHECK-LABEL: fn main(
    // CHECK: debug x => const 2_u32;
    // CHECK: assert(!const false,
    let x: u32 = 1 + 1;
}
