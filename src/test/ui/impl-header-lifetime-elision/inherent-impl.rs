// compile-pass

struct Foo<'a>(&'a u8);

impl Foo<'_> {
    fn x() {}
}

fn main() {}
