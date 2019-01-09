#![feature(nll)]

struct Bar;

struct Foo<'s> {
    bar: &'s mut Bar,
}

impl Foo<'_> {
    fn new(bar: &mut Bar) -> Self {
        Foo { bar } //~ ERROR unsatisfied lifetime constraints
    }
}

fn main() { }
