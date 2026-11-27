macro_rules! y {
    () => {
        N
    };
}

struct A<const N: usize>;

fn foo<const N: usize>() -> A<{ y!() }> {
    A::<1>
    //~^ ERROR: mismatched types
}

fn main() {}
