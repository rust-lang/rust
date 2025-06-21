// skip-filecheck
// EMIT_MIR_FOR_EACH_PANIC_STRATEGY
//! Tests that in lower opt levels we remove (more) storage statements using a simpler strategy.
//@ test-mir-pass: CopyProp
//@ compile-flags: -Copt-level=0

// EMIT_MIR issue_141649_debug.main.CopyProp.diff
fn main() {
    struct S(usize, usize);
    {
        let s1 = S(1, 2);
        drop(s1);
    }
    {
        let s2 = S(3, 4);
        drop(s2);
    }

    #[derive(Clone, Copy)]
    struct C(usize, usize);
    {
        let c1 = C(1, 2);
        drop(c1);
    }
    {
        let c2 = C(3, 4);
        drop(c2);
    }
}
