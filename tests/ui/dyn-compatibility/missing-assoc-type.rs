trait Foo {
    type Bar<T>;
}

fn bar(x: &dyn Foo) {} //~ ERROR the trait `Foo` is not dyn compatible

fn main() {}
