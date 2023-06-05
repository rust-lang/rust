// edition: 2021

#![feature(return_type_notation, async_fn_in_trait)]
//~^ WARN the feature `return_type_notation` is incomplete

trait Trait {
    async fn method() {}
}

fn foo<T: Trait<method(i32): Send>>() {}
//~^ ERROR argument types not allowed with return type notation
//~| ERROR associated type bounds are unstable

fn bar<T: Trait<method() -> (): Send>>() {}
//~^ ERROR return type not allowed with return type notation
//~| ERROR associated type bounds are unstable

fn baz<T: Trait<method(..): Send>>() {}
//~^ ERROR return type notation uses `()` instead of `(..)` for elided arguments

fn main() {}
