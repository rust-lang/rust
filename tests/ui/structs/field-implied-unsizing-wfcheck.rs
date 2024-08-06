struct Foo {
    nested: &'static Bar<dyn std::fmt::Debug>,
    //~^ ERROR the size for values of type `(dyn Debug + 'static)` cannot be known at compilation time
}

struct Bar<T>(T);

fn main() {
    let x = Foo { nested: &Bar(4) };
    //~^ ERROR the size for values of type `(dyn Debug + 'static)` cannot be known at compilation time
}
