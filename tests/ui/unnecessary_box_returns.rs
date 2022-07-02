#![warn(clippy::unnecessary_box_returns)]

struct Foo {}

// lint
fn boxed_usize() -> Box<usize> {
    Box::new(5)
}

// lint
fn boxed_foo() -> Box<Foo> {
    Box::new(Foo {})
}

// don't lint: str is unsized
fn boxed_str() -> Box<str> {
    "Hello, world!".to_string().into_boxed_str()
}

// don't lint: this has an unspecified return type
fn default() {}

// don't lint: this doesn't return a Box
fn string() -> String {
    String::from("Hello, world")
}

fn main() {
    // don't lint: this is a closure
    let a = || -> Box<usize> { Box::new(5) };
}
