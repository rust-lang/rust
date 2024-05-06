//@ check-pass

// See https://doc.rust-lang.org/1.77.0/nightly-rustc/rustc_lint/opaque_hidden_inferred_bound/static.OPAQUE_HIDDEN_INFERRED_BOUND.html#example

#![feature(type_alias_impl_trait)]
#![allow(dead_code)]

trait Duh {}

impl Duh for i32 {}

trait Trait {
    type Assoc: Duh;
}

impl<R: Duh, F: FnMut() -> R> Trait for F {
    type Assoc = R;
}

type Sendable = impl Send;

type Foo = impl Trait<Assoc = Sendable>;
                   //~^ WARNING opaque type `Foo` does not satisfy its associated type bounds

fn foo() -> Foo {
    || 42
}

fn main() {}
