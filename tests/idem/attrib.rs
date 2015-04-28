// Test attributes and doc comments are preserved.

extern crate Foo;
#[Attr1]
extern crate Bar;
#[Attr2]
#[Attr2]
extern crate Baz;

/// Blah blah blah.
impl Bar {
    /// Blah blah blooo.
    #[an_attribute]
    fn foo(&mut self) -> isize {}

    /// Blah blah bing.
    pub fn f2(self) {
        (foo, bar)
    }

    #[another_attribute]
    fn f3(self) -> Dog {
    }
}

/// Blah
fn main() {
    println!("Hello world!");
}
