// EMIT_MIR_FOR_EACH_PANIC_STRATEGY
//! Tests that we can propagate into places that are projections into unions
//@ compile-flags: -Zunsound-mir-opts -C debuginfo=full
fn val() -> u32 {
    1
}

// EMIT_MIR union.main.DestinationPropagation.diff
fn main() {
    // CHECK-LABEL: fn main(
    // CHECK: debug un => const Un
    // CHECK: debug _x => const 1_u32;
    // CHECK: bb0: {
    // CHECK-NEXT: return;
    union Un {
        us: u32,
    }

    let un = Un { us: val() };

    drop(unsafe { un.us });
}
