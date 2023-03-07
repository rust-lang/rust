// build-pass (FIXME(62277): could be check-pass?)

struct Foo<'a>(&'a u8);

impl Foo<'_> {
    fn x() {}
}

fn main() {}
