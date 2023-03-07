// Regression test for issue #55001. Previously, we would incorrectly
// cache certain trait selection results when checking for blanket impls,
// resulting in an ICE when we tried to confirm the cached ParamCandidate
// against an obligation.

pub struct DefaultAllocator;
pub struct Standard;
pub struct Inner;

pub trait Rand {}

pub trait Distribution<T> {}
pub trait Allocator<N> {}

impl<T> Rand for T where Standard: Distribution<T> {}

impl<A> Distribution<Point<A>> for Standard
where
DefaultAllocator: Allocator<A>,
Standard: Distribution<A> {}

impl Distribution<Inner> for Standard {}


pub struct Point<N>
where DefaultAllocator: Allocator<N>
{
    field: N
}

fn main() {}
