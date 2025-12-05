fn foo() -> unsafe<'a> &'a () {}

struct Foo {
    x: unsafe<'a> &'a (),
}

struct Bar(unsafe<'a> &'a ());

impl Trait for unsafe<'a> &'a () {}

fn empty() -> unsafe<> () {}
