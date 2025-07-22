//@ only-x86_64
//@ compile-flags: -C opt-level=3 -C target-cpu=x86-64
#![crate_type = "lib"]

/// Check that the fast-path of `is_ascii` uses a `pmovmskb` instruction.
/// Platforms lacking an equivalent instruction use other techniques for
/// optimizing `is_ascii`.
// CHECK-LABEL: @is_ascii_autovectorized
#[no_mangle]
pub fn is_ascii_autovectorized(s: &[u8]) -> bool {
    // CHECK: load <32 x i8>
    // CHECK-NEXT: icmp slt <32 x i8>
    // CHECK-NEXT: bitcast <32 x i1>
    // CHECK-NEXT: icmp eq i32
    s.is_ascii()
}
