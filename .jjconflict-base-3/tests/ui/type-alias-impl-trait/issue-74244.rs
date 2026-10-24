#![feature(type_alias_impl_trait)]

trait Allocator {
    type Buffer;
}

struct DefaultAllocator;

impl<T> Allocator for DefaultAllocator {
    //~^ ERROR: the type parameter `T` is not constrained
    type Buffer = ();
}

type A = impl Fn(<DefaultAllocator as Allocator>::Buffer);

#[define_opaque(A)]
fn foo() -> A {
    |_| ()
}

fn main() {}
