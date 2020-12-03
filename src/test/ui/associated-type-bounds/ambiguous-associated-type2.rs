// ignore-tidy-linelength

trait Foo {
    type Item;
}
trait Bar<T> {
    type Item;
}
trait Baz: Foo + Bar<Self::Item> {}
//~^ ERROR cycle detected when computing the super traits of `Baz` with associated type name `Item` [E0391]

fn main() {}
