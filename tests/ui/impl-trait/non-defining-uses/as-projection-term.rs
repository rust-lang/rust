//@ revisions: current next
//@[next] compile-flags: -Znext-solver
//@ ignore-compare-mode-next-solver (explicit revisions)
//@ check-pass

fn prove_proj<R>(_: impl FnOnce() -> R) {}
fn recur<'a>() -> impl Sized + 'a {
    // The closure has the signature `fn() -> opaque<'1>`. `prove_proj`
    // requires us to prove `<closure as FnOnce<()>>::Output = opaque<'2>`.
    // The old solver uses `replace_opaque_types_with_infer` during normalization
    // to replace `opaque<'2>` with its hidden type. If that hidden type is still an
    // inference variable at this point, we unify it with `opaque<'1>` and
    // end up ignoring that defining use as the hidden type is equal to its key.
    prove_proj(|| recur());
}

fn main() {}
