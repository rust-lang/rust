//@ check-pass

struct A<const M: u32> {}

struct B<const M: u32> {}

impl<const M: u32> B<M> {
    const M: u32 = M;
}

struct C<const M: u32> {
    a: A<{ B::<1>::M }>,
}

fn main() {}
