#[rustfmt::skip]
macro_rules! y {
    () => {
        { N }
        //~^ ERROR: generic parameters may not be used in const operations
    };
}

struct A<const N: usize>;

fn foo<const N: usize>() -> A<{ y!() }> {
    A::<1>
}

fn main() {}
