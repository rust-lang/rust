// issue #21405
// ignore-tidy-linelength

struct Foo;

fn foo<F>(f: F) where F: FnMut(Foo) {}

fn main() {
    foo(|s| s.is_empty());
    //~^ ERROR no method named `is_empty` found
}
