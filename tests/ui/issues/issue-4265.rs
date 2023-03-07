struct Foo {
      baz: usize
}

impl Foo {
    fn bar() {
        Foo { baz: 0 }.bar();
    }

    fn bar() { //~ ERROR duplicate definitions
    }
}

fn main() {}
