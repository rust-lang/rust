trait Foo {
    type Output<T>;

    fn baz();
}

enum Bar<T> {
    Simple {},
    Generic(T),
}

impl Foo for u8 {
    type Output<T> = Bar<T>;
    fn baz() {
        Self::Output::Simple {}; //~ ERROR type annotations needed
    }
}

fn main() {}
