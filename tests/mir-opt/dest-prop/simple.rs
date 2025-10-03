// EMIT_MIR_FOR_EACH_PANIC_STRATEGY
//! Copy of `nrvo-simple.rs`, to ensure that full dest-prop handles it too.
//@ test-mir-pass: DestinationPropagation
// EMIT_MIR simple.nrvo.DestinationPropagation.diff
fn nrvo(init: fn(&mut [u8; 1024])) -> [u8; 1024] {
    // CHECK-LABEL: fn nrvo(
    // CHECK: debug init => [[init:_.*]];
    // CHECK: debug buf => _0;
    // CHECK: _0 = [const 0_u8; 1024];
    // CHECK-NOT: {{_.*}} = copy [[init]];
    // CHECK: {{_.*}} = &mut _0;
    // CHECK: move [[init]](move {{_.*}})
    let mut buf = [0; 1024];
    init(&mut buf);
    buf
}

fn main() {
    let _ = nrvo(|buf| {
        buf[4] = 4;
    });
}
