struct Foo<const N: usize>;

impl<const N: usize = 1> Foo<N> {}
//~^ ERROR defaults for generic parameters are not allowed here

fn main() {}
