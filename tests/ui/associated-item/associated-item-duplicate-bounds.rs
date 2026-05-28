trait Adapter {
    const LINKS: usize;
}

struct Foo<A: Adapter> {
    adapter: A,
    links: [u32; A::LINKS], // Shouldn't suggest bounds already there.
    //~^ ERROR generic parameters may not be used in const operations
}

fn main() {}
