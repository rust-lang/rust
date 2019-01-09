struct Foo {
    field: i32,
}

impl Foo {
    fn foo<'a>(&self, x: &'a Foo) -> &'a Foo {

        if true { x } else { self } //~ ERROR lifetime mismatch

    }
}

fn main() {}
