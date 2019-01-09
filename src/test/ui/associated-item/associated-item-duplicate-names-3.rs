//
// Before the introduction of the "duplicate associated type" error, the
// program below used to result in the "ambiguous associated type" error E0223,
// which is unexpected.

trait Foo {
    type Bar;
}

struct Baz;

impl Foo for Baz {
    type Bar = i16;
    type Bar = u16; //~ ERROR duplicate definitions
}

fn main() {
    let x: Baz::Bar = 5;
}
