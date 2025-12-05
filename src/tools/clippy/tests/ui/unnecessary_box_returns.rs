#![warn(clippy::unnecessary_box_returns)]
//@no-rustfix
trait Bar {
    // lint
    fn baz(&self) -> Box<usize>;
    //~^ unnecessary_box_returns
}

pub struct Foo {}

impl Bar for Foo {
    // don't lint: this is a problem with the trait, not the implementation
    fn baz(&self) -> Box<usize> {
        Box::new(42)
    }
}

impl Foo {
    fn baz(&self) -> Box<usize> {
        //~^ unnecessary_box_returns

        // lint
        Box::new(13)
    }
}

// lint
fn bxed_usize() -> Box<usize> {
    //~^ unnecessary_box_returns

    Box::new(5)
}

// lint
fn _bxed_foo() -> Box<Foo> {
    //~^ unnecessary_box_returns

    Box::new(Foo {})
}

// don't lint: this is exported
pub fn bxed_foo() -> Box<Foo> {
    Box::new(Foo {})
}

// don't lint: str is unsized
fn bxed_str() -> Box<str> {
    "Hello, world!".to_string().into_boxed_str()
}

// don't lint: function contains the word, "box"
fn boxed_usize() -> Box<usize> {
    Box::new(7)
}

// don't lint: this has an unspecified return type
fn default() {}

// don't lint: this doesn't return a Box
fn string() -> String {
    String::from("Hello, world")
}

struct Huge([u8; 500]);
struct HasHuge(Box<Huge>);

impl HasHuge {
    // don't lint: The size of `Huge` is very large
    fn into_huge(self) -> Box<Huge> {
        self.0
    }
}

fn main() {
    // don't lint: this is a closure
    let a = || -> Box<usize> { Box::new(5) };
}
