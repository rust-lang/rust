//@ compile-flags: -Z mir-opt-level=0

// EMIT_MIR byte_slice.main.SimplifyCfg-pre-optimizations.after.mir
fn main() {
    // CHECK-LABEL: fn main(
    // CHECK: = const b"foo"
    // CHECK: = [const 5_u8, const 120_u8]
    let x = b"foo";
    let y = [5u8, b'x'];
}
