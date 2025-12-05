//@ compile-flags: -C no-prepopulate-passes -Copt-level=0

#![crate_type = "lib"]

#[repr(u16)]
pub enum E2 {
    A = 13,
    B = 42,
}

// For unoptimized code we produce a `br` instead of a `switch`. Compare with
// `tests/codegen/match-optimized.rs`

// CHECK-LABEL: @exhaustive_match_2
#[no_mangle]
pub fn exhaustive_match_2(e: E2) -> u8 {
    // CHECK: %[[CMP:.+]] = icmp eq i16 %{{.+}}, 13
    // CHECK-NEXT: br i1 %[[CMP:.+]],
    match e {
        E2::A => 0,
        E2::B => 1,
    }
}
