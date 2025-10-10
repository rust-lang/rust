//! Regression test for: <https://github.com/rust-lang/rust/issues/144132>
//!                      <https://github.com/rust-lang/rust/issues/135017>

struct Baz<const N: usize>(i32);

fn main() {
    let _: Baz<#[cfg(any())]> = todo!();
    //~^ ERROR attributes cannot be applied here
}

fn f(_param: #[attr]) {}
//~^ ERROR attributes cannot be applied to a function parameter's type
//~| ERROR expected type, found `)`

fn g() -> #[attr] { 0 }
//~^ ERROR attributes cannot be applied here

struct S {
    field: #[attr],
    //~^ ERROR attributes cannot be applied here
    field1: (#[attr], i32),
    //~^ ERROR attributes cannot be applied here
}

type Tuple = (#[attr], String);
//~^ ERROR attributes cannot be applied here

impl #[attr] {}
//~^ ERROR attributes cannot be applied here
