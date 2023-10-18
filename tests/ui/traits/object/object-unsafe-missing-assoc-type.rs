trait Foo {
    type Bar<T>;
}

fn bar(x: &dyn Foo) {} //~ ERROR the trait `Foo` cannot be made into an object

fn main() {}
