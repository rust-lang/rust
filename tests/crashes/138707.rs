//@ known-bug: #138707
//@edition:2024
//@compile-flags: --crate-type lib
use core::marker::PhantomData;

struct LeftReflector<S> {
    _phantom: PhantomData<S>,
}

struct DefaultAllocator {}

trait Allocator<R> {
    type Buffer;
}

struct U2 {}

impl Allocator<U2> for DefaultAllocator {
    type Buffer = [u8; 2];
}

impl<R> From<R> for LeftReflector<<DefaultAllocator as Allocator<R>>::Buffer>
where
    DefaultAllocator: Allocator<R>,
{
    fn from(_: R) -> Self {
        todo!()
    }
}

fn ice<D>(a: U2)
where
    DefaultAllocator: Allocator<D>,
{
    // ICE
    let _ = LeftReflector::from(a);
}
