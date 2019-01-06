// compile-flags: -Z continue-parse-after-error

trait Foo {
}

struct Bar;

impl Foo + Owned for Bar { //~ ERROR expected a trait, found type
}

fn main() { }
