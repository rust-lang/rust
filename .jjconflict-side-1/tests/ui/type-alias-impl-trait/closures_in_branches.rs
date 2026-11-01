#![feature(type_alias_impl_trait)]

type Foo = impl std::ops::FnOnce(String) -> usize;

#[define_opaque(Foo)]
fn foo(b: bool) -> Foo {
    if b {
        |x| x.len() //~ ERROR type annotations needed
    } else {
        panic!()
    }
}

type Foo1 = impl std::ops::FnOnce(String) -> usize;
#[define_opaque(Foo1)]
fn foo1(b: bool) -> Foo1 {
    |x| x.len()
}

fn bar(b: bool) -> impl std::ops::FnOnce(String) -> usize {
    if b {
        |x| x.len() //~ ERROR type annotations needed
    } else {
        panic!()
    }
}

fn bar1(b: bool) -> impl std::ops::FnOnce(String) -> usize {
    |x| x.len()
}

fn main() {}
