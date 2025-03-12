//@ edition:2021
//@ run-rustfix

#![allow(dead_code)]
trait Trait {
    async fn foo();

    async fn bar() -> i32;

    fn test(&self) -> impl Sized + '_;

    async fn baz(&self) -> &i32;
}

struct S;

impl Trait for S {}
//~^ ERROR not all trait items implemented

fn main() {}
