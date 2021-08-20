#![feature(type_alias_impl_trait)]
// check-pass
fn main() {}

// two definitions with different types
type Foo = impl std::fmt::Debug;

fn foo() -> Foo {
    ""
}

fn bar() -> Foo {
    panic!()
}

fn boo() -> Foo {
    loop {}
}
