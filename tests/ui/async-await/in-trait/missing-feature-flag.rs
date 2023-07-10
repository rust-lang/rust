// edition:2018
// [next] compile-flags: -Zlower-impl-trait-in-trait-to-assoc-ty
// revisions: current next

#![feature(async_fn_in_trait)]
#![feature(min_specialization)]

struct MyStruct;

trait MyTrait<T> {
    async fn foo(_: T) -> &'static str;
}

impl<T> MyTrait<T> for MyStruct {}
//~^ ERROR: not all trait items implemented, missing: `foo` [E0046]

impl MyTrait<i32> for MyStruct {
    async fn foo(_: i32) -> &'static str {}
    //~^ ERROR: `foo` specializes an item from a parent `impl`, but that item is not marked `default` [E0520]
    //~| ERROR: mismatched types [E0308]
}

fn main() {}
