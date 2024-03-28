// skip-filecheck
// EMIT_MIR_FOR_EACH_PANIC_STRATEGY
//! Tests that we can propagate into places that are projections into unions
//@ compile-flags: -Zunsound-mir-opts
//@ unit-test: DestinationPropagation
fn val() -> u32 {
    1
}

// EMIT_MIR union.main.DestinationPropagation.diff
fn main() {
    union Un {
        us: u32,
    }

    let un = Un { us: val() };

    drop(unsafe { un.us });
}
