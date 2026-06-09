//@ check-pass

// Regression test for an ICE: https://github.com/rust-lang/rust/issues/119916

#![feature(impl_trait_in_assoc_type)]
#![feature(type_alias_impl_trait)]

// `impl_trait_in_assoc_type` example from the bug report.
pub trait StreamConsumer {
    type BarrierStream;
    fn execute() -> Self::BarrierStream;
}

pub struct DispatchExecutor;

impl StreamConsumer for DispatchExecutor {
    type BarrierStream = impl Sized;
    fn execute() -> Self::BarrierStream {
        || -> _ {}
    }
}

// Functions that constrain TAITs can contain closures with an `_` in the return type.
type Foo = impl Sized;
#[define_opaque(Foo)]
fn foo() -> Foo {
    || -> _ {}
}

// The `_` in the closure return type can also be the TAIT itself.
type Bar = impl Sized;
#[define_opaque(Bar)]
fn bar() -> impl FnOnce() -> Bar {
    || -> _ {}
}

fn main() {}
