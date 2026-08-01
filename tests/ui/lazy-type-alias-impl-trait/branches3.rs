#![feature(type_alias_impl_trait)]

type Foo = impl for<'a> FnOnce(&'a str) -> usize;
type Bar = impl FnOnce(&'static str) -> usize;

#[define_opaque(Foo)]
fn foo() -> Foo {
    if true {
        |s| s.len()
        //~^ ERROR implementation of `FnOnce` is not general enough
        //~| ERROR implementation of `FnOnce` is not general enough
    } else {
        panic!()
    }
}

#[define_opaque(Bar)]
fn bar() -> Bar {
    if true {
        |s| s.len()
    } else {
        panic!()
    }
}

fn foo2() -> impl for<'a> FnOnce(&'a str) -> usize {
    if true {
        |s| s.len()
        //~^ ERROR implementation of `FnOnce` is not general enough
        //~| ERROR implementation of `FnOnce` is not general enough
    } else {
        panic!()
    }
}
fn bar2() -> impl FnOnce(&'static str) -> usize {
    if true {
        |s| s.len()
    } else {
        panic!()
    }
}

fn main() {}
