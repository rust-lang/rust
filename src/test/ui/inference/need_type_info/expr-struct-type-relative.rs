// regression test for #98598

trait Foo {
    type Output;

    fn baz() -> Self::Output;
}

fn needs_infer<T>() {}

struct Bar {}

impl Foo for u8 {
    type Output = Bar;
    fn baz() -> Self::Output {
        needs_infer(); //~ ERROR type annotations needed
        Self::Output {}
    }
}

fn main() {}
