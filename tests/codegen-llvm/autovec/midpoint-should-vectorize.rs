//@ revisions: X86_64
//@ compile-flags: -Copt-level=3
//@[X86_64] only-x86_64
//@[X86_64] compile-flags: -Ctarget-feature=+avx2

#![crate_type = "lib"]
#![no_std]

// Ensure u64::midpoint can auto-vectorize

// CHECK-LABEL: define{{.*}}void @midpoint_vec(
// CHECK: load <4 x i64>
// CHECK-NEXT: load <4 x i64>
// CHECK-NEXT: xor <4 x i64>
// CHECK-NEXT: lshr <4 x i64>
// CHECK-NEXT: and <4 x i64>
// CHECK-NEXT: add <4 x i64>
// CHECK-NEXT: store <4 x i64>
// CHECK-NEXT: ret void
#[unsafe(no_mangle)]
pub fn midpoint_vec(r: &mut [u64; 4], a: &[u64; 4], b: &[u64; 4]) {
    for i in 0..4 {
        r[i] = a[i].midpoint(b[i]);
    }
}
