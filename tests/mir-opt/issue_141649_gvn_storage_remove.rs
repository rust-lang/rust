// skip-filecheck
// EMIT_MIR_FOR_EACH_PANIC_STRATEGY
//@ test-mir-pass: GVN

// EMIT_MIR issue_141649_gvn_storage_remove.main.GVN.diff
fn main() {
    struct S(u32, u32);
    {
        let s1 = S(1, 2);
        drop(s1);
    }
    {
        let s2 = S(3, 4);
        drop(s2);
    }

    #[derive(Clone, Copy)]
    struct C(u32, u32);
    {
        let c1 = C(1, 2);
        drop(c1);
    }
    {
        let c2 = C(3, 4);
        drop(c2);
    }
}
