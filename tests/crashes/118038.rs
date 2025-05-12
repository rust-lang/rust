//@ known-bug: #118038
#![feature(non_lifetime_binders)]

fn trivial<A>()
where
    for<B> dyn Fn(A, *const A): Fn(A, *const B),
{
}

fn main() {
    trivial::<u8>();
}
