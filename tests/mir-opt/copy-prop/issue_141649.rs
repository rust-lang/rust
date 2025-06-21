// skip-filecheck
// EMIT_MIR_FOR_EACH_PANIC_STRATEGY
//@ test-mir-pass: CopyProp

// EMIT_MIR issue_141649.main.CopyProp.diff
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
    {
        let s3 = S(5, 6);
        let borrowed_s3 = &s3;
        opaque(borrowed_s3);
        drop(s3);
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

#[inline(never)]
fn opaque<T>(a: T) -> T {
    a
}
