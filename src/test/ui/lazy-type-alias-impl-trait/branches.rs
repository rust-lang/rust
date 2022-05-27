#![feature(type_alias_impl_trait)]

// check-pass

type Foo = impl std::fmt::Debug;

fn foo(b: bool) -> Foo {
    if b {
        vec![42_i32]
    } else {
        std::iter::empty().collect()
    }
}

fn main() {}
