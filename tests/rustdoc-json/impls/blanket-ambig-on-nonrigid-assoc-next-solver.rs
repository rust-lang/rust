//@ compile-flags: -Znext-solver

// Regression test for <https://github.com/rust-lang/rust/issues/162557>

pub trait Service<Request> {
    type Future;
}

pub trait ZebraService<Request>: Service<Request> {}

impl<MaybeVerify, Request> ZebraService<Request> for MaybeVerify where
    MaybeVerify: Service<Request, Future: 'static>
{
}

pub struct Verifier;

impl Service<()> for Verifier {
    type Future = &'static ();
}

//@ set blanket = "$.index[?(@.inner.impl.blanket_impl.generic=='MaybeVerify')].id"
//@ has "$.index[?(@.name=='Verifier')].inner.struct.impls[*]" $blanket
