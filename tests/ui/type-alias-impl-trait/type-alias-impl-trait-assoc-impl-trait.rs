// check-pass

#![feature(type_alias_impl_trait)]
#![allow(dead_code)]

type Foo = impl Iterator<Item = impl Send>;

fn make_foo() -> Foo {
    vec![1, 2].into_iter()
}

type Bar = impl Send;
type Baz = impl Iterator<Item = Bar>;

fn make_baz() -> Baz {
    vec!["1", "2"].into_iter()
}

fn main() {}
