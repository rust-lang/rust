//! Regression test for: <https://github.com/rust-lang/rust/issues/144132>
//!                      <https://github.com/rust-lang/rust/issues/135017>

//@ run-rustfix

#![allow(dead_code, unused_variables)]

struct Foo<T>(T);
struct Bar<'a>(&'a i32);
struct Baz<const N: usize>(i32);

fn main() {
    let foo: Foo<#[cfg(not(wrong))] i32> = Foo(2i32);
    //~^ ERROR attributes cannot be applied to generic arguments

    let _: #[attr] &'static str = "123";
    //~^ ERROR attributes cannot be applied to types

    let _: Bar<#[cfg(any())] 'static> = Bar(&123);
    //~^ ERROR attributes cannot be applied to generic arguments

    let _: Baz<#[cfg(any())] 42> = Baz(42);
    //~^ ERROR attributes cannot be applied to generic arguments

    let _: Foo<#[cfg(not(wrong))]String> = Foo(String::new());
    //~^ ERROR attributes cannot be applied to generic arguments

    let _: Bar<#[cfg(any())]       'static> = Bar(&456);
    //~^ ERROR attributes cannot be applied to generic arguments

    let _generic: Box<#[attr] i32> = Box::new(1);
    //~^ ERROR attributes cannot be applied to generic arguments

    let _assignment: #[attr] i32 = *Box::new(1);
    //~^ ERROR attributes cannot be applied to types

    let _complex: Vec<#[derive(Debug)] String> = vec![];
    //~^ ERROR attributes cannot be applied to generic arguments

    let _nested: Box<Vec<#[cfg(feature = "test")] u64>> = Box::new(vec![]);
    //~^ ERROR attributes cannot be applied to generic arguments
}

fn g() -> #[attr] i32 { 0 }
//~^ ERROR attributes cannot be applied to types

struct S {
    field: #[attr] i32,
    //~^ ERROR attributes cannot be applied to types
    field1: (#[attr] i32, i32),
    //~^ ERROR attributes cannot be applied to types
}

type Tuple = (#[attr] i32, String);
//~^ ERROR attributes cannot be applied to types

impl #[attr] S {}
//~^ ERROR attributes cannot be applied to types
