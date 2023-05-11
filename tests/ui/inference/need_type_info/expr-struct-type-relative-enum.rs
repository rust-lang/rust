trait Foo {
    type Output;

    fn baz() -> Self::Output;
}

fn needs_infer<T>() {}

enum Bar {
    Variant {}
}

impl Foo for u8 {
    type Output = Bar;
    fn baz() -> Self::Output {
        needs_infer(); //~ ERROR type annotations needed
        Self::Output::Variant {}
    }
}

fn main() {}
