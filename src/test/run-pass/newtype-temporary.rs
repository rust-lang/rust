#[derive(PartialEq, Debug)]
struct Foo(usize);

fn foo() -> Foo {
    Foo(42)
}

pub fn main() {
    assert_eq!(foo(), Foo(42));
}
