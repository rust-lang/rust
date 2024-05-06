// skip-filecheck
//@ compile-flags: -Z mir-opt-level=0

// EMIT_MIR byte_slice.main.SimplifyCfg-pre-optimizations.after.mir
fn main() {
    let x = b"foo";
    let y = [5u8, b'x'];
}
