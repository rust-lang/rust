macro_rules! empty {
    () => {};
}

struct Foo<const N: usize>;

#[rustfmt::skip]
impl Foo<{ empty!{}; }> {}
//~^ ERROR: mismatched types
#[rustfmt::skip]
impl Foo<{ empty!(); }> {}
//~^ ERROR: mismatched types

fn main() {}
