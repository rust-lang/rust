//@ revisions: S Z
//@[S] compile-flags: -Copt-level=s
//@[Z] compile-flags: -Copt-level=z
//@ edition: 2021

// `-Copt-level=s`/`z` derive a MIR opt level of 2, and MIR inlining must run there. LLVM's own
// inliner refuses the callee below at both levels, so "is `work` still called?" is a probe for
// whether the *MIR* inliner ran.

#![crate_type = "lib"]

pub fn work(mut x: u32, n: u32) -> u32 {
    let mut acc = 0u32;
    let mut k = 0u32;
    for i in 0..n {
        x = x.wrapping_mul(1664525).wrapping_add(1013904223);
        x ^= x >> 13;
        acc = acc.wrapping_add(x).wrapping_add(i);
        if x & 1 == 0 {
            acc ^= acc.rotate_left(7);
        } else {
            acc = acc.wrapping_sub(3);
            k = k.wrapping_add(x >> 3);
        }
        acc = acc.rotate_right(5);
        match x % 5 {
            0 => acc = acc.wrapping_add(11),
            1 => acc ^= 0x5bf03635,
            2 => acc = acc.wrapping_mul(3),
            3 => acc = acc.rotate_left(9),
            _ => acc = acc.wrapping_sub(k),
        }
    }
    acc ^ x ^ k
}

#[inline(never)]
pub fn sink(x: u32) -> u32 {
    core::hint::black_box(x)
}

// CHECK-LABEL: @caller(
#[no_mangle]
pub fn caller(x: u32) -> u32 {
    // CHECK-NOT: call {{.*}}work
    // CHECK: call {{.*}}sink
    sink(work(x, 64))
}
