struct A<const N: usize>;

#[rustfmt::skip]
fn foo<const N: usize>() -> A<{ { N } }> {
    //~^ ERROR: generic parameters may not be used in const operations
    A::<1>
}

fn main() {}
