// Test that unsupported uses of `Self` in impls don't crash

struct Bar;

trait Foo {
    type Baz;
}

trait SuperFoo {
    type SuperBaz;
}

impl Foo for Bar {
    type Baz = bool;
}

impl SuperFoo for Bar {
    type SuperBaz = bool;
}

impl Bar {
    fn f() {
        let _: <Self>::Baz = true;
        //~^ ERROR ambiguous associated type
        let _: Self::Baz = true;
        //~^ ERROR ambiguous associated type
    }
}

fn main() {}
