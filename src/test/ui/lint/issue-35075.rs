struct Bar<T> {
    inner: Foo<T> //~ ERROR cannot find type `Foo` in this scope
}

enum Baz<T> {
    Foo(Foo<T>) //~ ERROR cannot find type `Foo` in this scope
}

fn main() {}
