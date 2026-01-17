//@ only-loongarch64
//@ compile-flags: -C opt-level=3
#![crate_type = "lib"]

/// Check that the fast-path of `is_ascii` uses a `vmskltz.b` instruction.
/// Platforms lacking an equivalent instruction use other techniques for
/// optimizing `is_ascii`.
///
/// Note: x86_64 uses explicit SSE2 intrinsics instead of relying on
/// auto-vectorization. See `slice-is-ascii-avx512.rs`.
// CHECK-LABEL: @is_ascii_autovectorized
#[no_mangle]
pub fn is_ascii_autovectorized(s: &[u8]) -> bool {
    // CHECK: load <32 x i8>
    // CHECK-NEXT: icmp slt <32 x i8>
    // CHECK-NEXT: bitcast <32 x i1>
    // CHECK-NEXT: icmp eq i32
    s.is_ascii()
}
