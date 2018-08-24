struct Foo;

impl Foo {
    fn bar(self) {}

    fn foo() {
        self.bar(); //~ ERROR E0424
    }
}

fn main () {
}
