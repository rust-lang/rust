//@ known-bug: #121963
#![feature(generic_const_exprs)]
use std::marker::PhantomData;

trait Arch {
    const CHANNEL_COUNT: usize = 2;
}

struct Channel<const N: usize> {
    r: [u8; N],
}

struct Dram<A: Arch, S = Channel<{ A::CHANNEL_COUNT }>> {
    a: PhantomData<A>,
    s: PhantomData<S>,
}

struct C<A: Arch>
where
    Channel<{ A::CHANNEL_COUNT }, u8>: Sized,
{
    b: Dram<A>,
    //  b: Dram<A, Channel<{ A::CHANNEL_COUNT }>>,  // When I specified generic here, it worked
}

fn main() {}
