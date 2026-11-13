//@ revisions: O2 S Z
//@[O2] compile-flags: -Copt-level=2
//@[S] compile-flags: -Copt-level=s
//@[Z] compile-flags: -Copt-level=z
//@ only-x86_64

// Loop vectorization is enabled only for the speed objective, so both `-Copt-level=s` and `z` leave
// it off. The loop below is long enough that LLVM at `-O2` does vectorize it.

#![crate_type = "lib"]

// CHECK-LABEL: @scale_add(
// O2: fmul <4 x float>
// O2: fadd <4 x float>
// O2: store <4 x float>
// S-NOT: <4 x float>
// Z-NOT: <4 x float>
#[no_mangle]
pub fn scale_add(a: &[f32], b: &[f32], c: &mut [f32]) {
    for i in 0..64 {
        c[i] = a[i] * b[i] + 3.0;
    }
}
