#![feature(type_alias_impl_trait)]

type Foo = impl std::fmt::Debug;

fn foo(b: bool) -> Foo {
    if b {
        return 42
    }
    let x: u32 = foo(false) + 42; //~ ERROR cannot add
    99
}

fn bar(b: bool) -> impl std::fmt::Debug {
    if b {
        return 42
    }
    let x: u32 = bar(false) + 42; //~ ERROR cannot add
    99
}

fn main() {}
