trait Foo1 {}

trait Foo2 {
    type Bar;
}

trait Foo3 {
    type Bar;
}

impl<T> Foo1 for u32 where T: Clone {}
//~^ ERROR: the type parameter `T` is not constrained

impl<T, U> Foo1 for String where T: Iterator<Item = U> {}
//~^ ERROR: the type parameter `T` is not constrained
//~| ERROR: the type parameter `U` is not constrained

impl<T, U> Foo2 for T where T: Foo2<Bar = U> { type Bar = U; }
//~^ ERROR: the type parameter `U` is not constrained

impl<T, U> Foo3 for T where T: Foo3<Bar = Option<U>> { type Bar = Option<U>; }
//~^ ERROR: the type parameter `U` is not constrained

fn main() {}
