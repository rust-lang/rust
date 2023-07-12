trait Foo {
    type Item;
}
trait Bar<T> {
    type Item;
}
trait Baz: Foo + Bar<Self::Item> {}
//~^ ERROR cycle detected when computing the super traits of `Baz` with associated type name `Item`. see https://rustc-dev-guide.rust-lang.org/overview.html#queries and https://rustc-dev-guide.rust-lang.org/query.html for more information. [E0391]


fn main() {}
