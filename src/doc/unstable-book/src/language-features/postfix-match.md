# `postfix-match`

`postfix-match` adds the feature for matching upon values postfix
the expressions that generate the values.

```rust,edition2021
#![feature(postfix_match)]

enum Foo {
    Bar,
    Baz
}

fn get_foo() -> Foo {
    Foo::Bar
}

get_foo().match {
    Foo::Bar => {},
    Foo::Baz => panic!(),
}
```
