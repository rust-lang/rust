struct Foo;

impl Foo {
    fn bar(&self) {}

    fn foo(&self) {
        self.bar(); //~ ERROR E0424
    }

    fn baz(&self, _: i32) {
        self.bar(); //~ ERROR E0424
    }

    fn qux(&self) {
        let _ = || self.bar(); //~ ERROR E0424
    }
}

fn main () {
    let my_self = "self"; //~ ERROR E0424
}
