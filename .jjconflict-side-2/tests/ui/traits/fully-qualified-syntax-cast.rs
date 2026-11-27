// Regression test for #98565: Provide diagnostics when the user uses
// the built-in type `str` in a cast where a trait is expected.

trait Foo {
    fn foo(&self);
}

impl Foo for String {
    fn foo(&self) {
        <Self as str>::trim(self);
        //~^ ERROR expected trait, found builtin type `str`
    }
}

fn main() {}
