trait Foo1 {}

trait Foo2 {
    type Bar;
}

impl<const N: usize> Foo1 for u32 where [u32; N]: Clone {}
//~^ ERROR: the const parameter `N` is not constrained

impl<T, const N: usize> Foo1 for String where T: Iterator<Item = [u32; N]> {}
//~^ ERROR: the type parameter `T` is not constrained
//~| ERROR: the const parameter `N` is not constrained

impl<T, const N: usize> Foo2 for T where T: Foo2<Bar = [u32; N]> { type Bar = [u32; N]; }
//~^ ERROR: the const parameter `N` is not constrained

fn main() {}
