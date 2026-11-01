struct Foo {
    field: i32,
}

impl Foo {
    fn foo<'a>(&self, x: &Foo) -> &Foo {
        if true { x } else { self }
        //~^ ERROR lifetime may not live long enough
    }
}

fn main() {}
