struct Bar<T> {
    inner: Foo<T> //~ ERROR cannot find type `Foo`
}

enum Baz<T> {
    Foo(Foo<T>) //~ ERROR cannot find type `Foo`
}

fn main() {}
