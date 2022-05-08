#![feature(type_alias_impl_trait)]

type Foo = impl std::fmt::Debug;

fn foo(b: bool) -> Foo {
    if b {
        vec![42_i32]
    } else {
        std::iter::empty().collect()
        //~^ ERROR `Foo` cannot be built from an iterator over elements of type `_`
    }
}

fn main() {}
