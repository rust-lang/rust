// check-pass

#![feature(type_alias_impl_trait)]
#![allow(dead_code)]

type Foo = Vec<impl Send>;

fn make_foo() -> Foo {
    vec![true, false]
}

fn main() {}
