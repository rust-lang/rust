//! Tests that we can propagate into places that are projections into unions
//@ test-mir-pass: GVN
//@ compile-flags: -Zinline-mir

fn val() -> u32 {
    1
}

// EMIT_MIR union.main.GVN.diff
fn main() {
    // CHECK-LABEL: fn main(
    // CHECK: debug un => [[un:_.*]];
    // CHECK: bb0: {
    // CHECK: [[un]] = const Un {{{{ us: 1_u32 }}}};
    union Un {
        us: u32,
    }

    let un = Un { us: val() };

    drop(unsafe { un.us });
}
