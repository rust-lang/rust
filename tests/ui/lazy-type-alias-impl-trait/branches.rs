// check-pass

#![feature(type_alias_impl_trait)]

type Foo = impl std::fmt::Debug;

fn foo(b: bool) -> Foo {
    if b {
        vec![42_i32]
    } else {
        std::iter::empty().collect()
    }
}

type Bar = impl std::fmt::Debug;

fn bar(b: bool) -> Bar {
    let x: Bar = if b {
        vec![42_i32]
    } else {
        std::iter::empty().collect()
    };
    x
}

fn main() {}
