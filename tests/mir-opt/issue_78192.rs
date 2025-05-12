// skip-filecheck
//@ compile-flags: -Zmir-opt-level=1 -Zinline-mir
pub fn f<T>(a: &T) -> *const T {
    let b: &*const T = &(a as *const T);
    *b
}

fn main() {
    f(&2);
}

// EMIT_MIR issue_78192.f.InstSimplify-after-simplifycfg.diff
