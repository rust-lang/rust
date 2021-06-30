// run-pass

#![feature(derive_default_enum)]

// nb: does not impl Default
#[derive(Debug, PartialEq)]
struct NotDefault;

#[derive(Debug, Default, PartialEq)]
enum Foo {
    #[default]
    Alpha,
    #[allow(dead_code)]
    Beta(NotDefault),
}

fn main() {
    assert_eq!(Foo::default(), Foo::Alpha);
}
