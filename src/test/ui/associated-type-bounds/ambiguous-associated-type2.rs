trait Foo {
    type Item;
}
trait Bar<T> {
    type Item;
}
trait Baz: Foo + Bar<Self::Item> {}
//~^ ERROR cycle detected when computing the supertraits of `Baz` [E0391]

fn main() {}
