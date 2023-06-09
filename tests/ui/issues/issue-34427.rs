// run-pass
// Issue #34427: On ARM, the code in `foo` at one time was generating
// a machine code instruction of the form: `str r0, [r0, rN]!` (for
// some N), which is not legal because the source register and base
// register cannot be identical in the preindexed form signalled by
// the `!`.
//
// See LLVM bug: https://llvm.org/bugs/show_bug.cgi?id=28809

#[inline(never)]
fn foo(n: usize) -> Vec<Option<(*mut (), &'static ())>> {
    (0..n).map(|_| None).collect()
}

fn main() {
    let _ = (foo(10), foo(32));
}
