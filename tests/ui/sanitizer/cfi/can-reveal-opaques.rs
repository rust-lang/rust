//@ needs-sanitizer-cfi
//@ compile-flags: -Ccodegen-units=1 -Clto -Ctarget-feature=-crt-static -Zsanitizer=cfi -C unsafe-allow-abi-mismatch=sanitizer
//@ no-prefer-dynamic
//@ only-x86_64-unknown-linux-gnu
//@ ignore-backends: gcc
//@ build-pass

// See comment below for why this test exists.

trait Tr<U> {
    type Projection;
}

impl<F, U> Tr<U> for F
where
    F: Fn() -> U
{
    type Projection = U;
}

fn test<B: Tr<U>, U>(b: B) -> B::Projection
{
    todo!()
}

fn main() {
    fn rpit_fn() -> impl Sized {}

    // When CFI runs, it tries to compute the signature of the call. This
    // ends up giving us a signature of:
    //     `fn test::<rpit_fn, ()>() -> <rpit_fn as Tr<()>>::Projection`,
    // where `rpit_fn` is the ZST FnDef for the function. However, we were
    // previously using a Reveal::UserFacing param-env. This means that the
    // `<rpit_fn as Tr<()>>::Projection` return type is impossible to normalize,
    // since it would require proving `rpit_fn: Fn() -> ()`, but we cannot
    // prove that the `impl Sized` opaque is `()` with a user-facing param-env.
    // This leads to a normalization error, and then an ICE.
    //
    // Side-note:
    // So why is the second generic of `test` "`()`", and not the
    // `impl Sized` since we inferred it from the return type of `rpit_fn`
    // during typeck? Well, that's because we're using the generics from the
    // terminator of the MIR, which has had the PostAnalysisNormalize pass performed on it.
    let _ = test(rpit_fn);
}
