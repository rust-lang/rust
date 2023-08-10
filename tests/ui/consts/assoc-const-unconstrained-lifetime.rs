struct Foo;

impl<'a> Foo {
    const CONST: &'a str = "";
    //~^ ERROR the lifetime parameter `'a` is not constrained by the impl trait, self type, or predicates
}

fn main() {}
