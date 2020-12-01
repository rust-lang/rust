// Test for for diagnostic improvement issue #75907

mod foo {
    pub(crate) struct Foo(u8);
    pub(crate) struct Bar(pub u8, u8, Foo);

    pub(crate) fn make_bar() -> Bar {
        Bar(1, 12, Foo(10))
    }
}

use foo::{make_bar, Bar, Foo};

fn main() {
    let Bar(x, y, Foo(z)) = make_bar();
    //~^ ERROR cannot match against a tuple struct which contains private fields
    //~| ERROR cannot match against a tuple struct which contains private fields
}
