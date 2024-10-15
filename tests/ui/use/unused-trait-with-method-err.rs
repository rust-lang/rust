// Test that we don't issue an unused import warning when there's
// a method lookup error and that trait was possibly applicable.

use foo::Bar;

mod foo {
    pub trait Bar {
        fn uwu(&self) {}
    }
}

struct Foo;

fn main() {
    Foo.uwu();
    //~^ ERROR no method named `uwu` found for struct `Foo` in the current scope
}
