//! Regression test for: <https://github.com/rust-lang/rust/issues/144132>
//!                      <https://github.com/rust-lang/rust/issues/135017>

struct Foo<T>(T);

fn main() {
    let foo: Foo<#[cfg(not(wrong))] String> = todo!();
    //~^ ERROR attributes cannot be applied to generic type arguments

    let _generic: Box<#[attr] i32> = Box::new(1);
    //~^ ERROR attributes cannot be applied to generic type arguments

    let _assignment: #[attr] i32 = Box::new(1);
    //~^ ERROR attributes cannot be applied to a type

    let _complex: Vec<#[derive(Debug)] String> = vec![];
    //~^ ERROR attributes cannot be applied to generic type arguments

    let _nested: Box<Vec<#[cfg(feature = "test")] u64>> = Box::new(vec![]);
    //~^ ERROR attributes cannot be applied to generic type arguments
}

fn f(_param: #[attr] i32) {}
//~^ ERROR attributes cannot be applied to a function parameter's type

fn g() -> #[attr] i32 { 0 }
//~^ ERROR attributes cannot be applied to a type

struct S {
    field: #[attr] i32,
    //~^ ERROR attributes cannot be applied to a type
    field1: (#[attr] i32, i32),
    //~^ ERROR attributes cannot be applied to a type
}

type Tuple = (#[attr] i32, String);
//~^ ERROR attributes cannot be applied to a type

impl #[attr] S {}
//~^ ERROR attributes cannot be applied to a type
