// edition:2021
// run-rustfix

#![feature(async_fn_in_trait, return_position_impl_trait_in_trait)]

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
