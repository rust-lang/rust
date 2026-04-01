//@ compile-flags: -Copt-level=3
#![crate_type = "lib"]

/// Ensure the function is properly optimized
/// In the issue #133528, the function was not getting optimized
/// whereas, a version with `bytes` wrapped into a `black_box` was optimized
/// It was probably a LLVM bug that was fixed in LLVM 20

// CHECK-LABEL: @looping_over_ne_bytes
// CHECK: icmp eq i64 %input, -1
// CHECK-NEXT: ret i1
#[no_mangle]
fn looping_over_ne_bytes(input: u64) -> bool {
    let bytes = input.to_ne_bytes();
    bytes.iter().all(|x| *x == !0)
}
