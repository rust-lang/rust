// revisions: base nll
// ignore-compare-mode-nll
//[nll] compile-flags: -Z borrowck=mir

struct Bar;

struct Foo<'s> {
    bar: &'s mut Bar,
}

impl Foo<'_> {
    fn new(bar: &mut Bar) -> Self {
        Foo { bar } //~ERROR
    }
}

fn main() { }
