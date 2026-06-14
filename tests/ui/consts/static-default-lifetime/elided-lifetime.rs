//@check-pass

struct Foo<'a>(&'a ());

impl Foo<'_> {
    const STATIC: &str = "";
}

trait Bar {
    const STATIC: &str;
}

impl Bar for Foo<'_> {
    const STATIC: &str = "";
}

fn main() {}
