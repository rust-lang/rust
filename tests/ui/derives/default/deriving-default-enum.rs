//@ run-pass

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

// #[default] on a generic enum does not add `Default` bounds to the type params.
#[derive(Default)]
enum MyOption<T> {
    #[default]
    None,
    #[allow(dead_code)]
    Some(T),
}

fn main() {
    assert!(matches!(Foo::default(), Foo::Alpha));
    assert!(matches!(MyOption::<NotDefault>::default(), MyOption::None));
}
