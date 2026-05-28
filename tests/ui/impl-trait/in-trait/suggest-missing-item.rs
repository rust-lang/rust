//@ edition:2021
//@ run-rustfix

#![allow(dead_code)]
trait Trait {
    #[allow(async_fn_in_trait)]
    async fn foo();

    #[allow(async_fn_in_trait)]
    async fn bar() -> i32;

    fn test(&self) -> impl Sized + '_;

    #[allow(async_fn_in_trait)]
    async fn baz(&self) -> &i32;
}

struct S;

impl Trait for S {}
//~^ ERROR not all trait items implemented

fn main() {}
