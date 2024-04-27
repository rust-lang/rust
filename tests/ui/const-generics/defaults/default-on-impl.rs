struct Foo<const N: usize>;

impl<const N: usize = 1> Foo<N> {}
//~^ ERROR defaults for const parameters are only allowed

fn main() {}
