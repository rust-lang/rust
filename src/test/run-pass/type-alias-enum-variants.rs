#![feature(type_alias_enum_variants)]

#[derive(Debug, PartialEq, Eq)]
enum Foo {
    Bar(i32),
    Baz { i: i32 },
}

type FooAlias = Foo;
type OptionAlias = Option<i32>;

impl Foo {
    fn foo() -> Self {
        Self::Bar(3)
    }
}

fn main() {
    let t = FooAlias::Bar(1);
    assert_eq!(t, Foo::Bar(1));
    let t = FooAlias::Baz { i: 2 };
    assert_eq!(t, Foo::Baz { i: 2 });
    match t {
        FooAlias::Bar(_i) => {}
        FooAlias::Baz { i } => { assert_eq!(i, 2); }
    }
    assert_eq!(Foo::foo(), Foo::Bar(3));

    assert_eq!(OptionAlias::Some(4), Option::Some(4));
}
