#![feature(min_const_generics)]
// FIXME(const_generics): should be stable soon

trait Adapter {
    const LINKS: usize;
}

struct Foo<A: Adapter> {
    adapter: A,
    links: [u32; A::LINKS],
    //~^ ERROR generic parameters may not be used
}

fn main() {}
