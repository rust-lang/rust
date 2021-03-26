// revisions: min_tait full_tait
#![feature(min_type_alias_impl_trait)]
#![cfg_attr(full_tait, feature(type_alias_impl_trait))]
//[full_tait]~^ WARN incomplete

trait Allocator {
    type Buffer;
}

struct DefaultAllocator;

impl<T> Allocator for DefaultAllocator {
    //~^ ERROR: the type parameter `T` is not constrained
    type Buffer = ();
}

type A = impl Fn(<DefaultAllocator as Allocator>::Buffer);

fn foo() -> A {
    |_| ()
}

fn main() {}
