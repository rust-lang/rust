//@ compile-flags: -Znormalize-docs

#![feature(type_alias_impl_trait)]

trait Allocator {
    type Buffer;
}

struct DefaultAllocator;

// This unconstrained impl parameter causes the normalization of
// `<DefaultAllocator as Allocator>::Buffer` to be ambiguous,
// which caused an ICE with `-Znormalize-docs`.
impl<T> Allocator for DefaultAllocator {
    //~^ ERROR: the type parameter `T` is not constrained
    type Buffer = ();
}

type A = impl Fn(<DefaultAllocator as Allocator>::Buffer);

fn foo() -> A {
    |_| ()
}

fn main() {}
