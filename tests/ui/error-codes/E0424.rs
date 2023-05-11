struct Foo;

impl Foo {
    fn bar(self) {}

    fn foo() {
        self.bar(); //~ ERROR E0424
    }

    fn baz(_: i32) {
        self.bar(); //~ ERROR E0424
    }

    fn qux() {
        let _ = || self.bar(); //~ ERROR E0424
    }
}

fn main () {
    let self = "self"; //~ ERROR E0424
}
