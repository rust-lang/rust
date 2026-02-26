//@ compile-flags: -Zmir-opt-level=1 -Zinline-mir
pub fn f<T>(a: &T) -> *const T {
    // CHECK-LABEL: fn f(
    // CHECK: &raw const (*_1)
    let b: &*const T = &(a as *const T);
    *b
}

fn main() {
    f(&2);
}

// EMIT_MIR issue_78192.f.InstSimplify-after-simplifycfg.diff
