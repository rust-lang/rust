//! Regression test for <https://github.com/rust-lang/rust/issues/4265>.
//! Test method which calls duplicate method on self doesn't ICE.

struct Foo {
      baz: usize
}

impl Foo {
    fn bar() {
        Foo { baz: 0 }.bar();
        //~^ ERROR: no method named `bar` found
    }

    fn bar() { //~ ERROR duplicate definitions
    }
}

fn main() {}
