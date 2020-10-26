// EMIT_MIR issue_78192.f.InstCombine.diff
pub fn f<T>(a: &T) -> *const T {
    let b: &*const T = &(a as *const T);
    *b
}

fn main() {
    f(&2);
}
