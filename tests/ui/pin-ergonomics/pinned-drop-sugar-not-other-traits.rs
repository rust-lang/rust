//@ edition:2024

#![feature(pin_ergonomics)]
#![allow(incomplete_features)]

use std::pin::Pin;

struct S;

trait NeedsPinDrop {
    fn pin_drop(self: Pin<&mut Self>);
}

impl NeedsPinDrop for S {
    //~^ ERROR not all trait items implemented, missing: `pin_drop` [E0046]
    fn drop(&pin mut self) {}
    //~^ ERROR method `drop` with `&pin mut self` is only supported for the `Drop` trait
}

trait HasDrop {
    fn drop(self: Pin<&mut Self>);
}

impl HasDrop for S {
    //~^ ERROR not all trait items implemented, missing: `drop` [E0046]
    fn drop(&pin mut self) {}
    //~^ ERROR method `drop` is not a member of trait `HasDrop` [E0407]
}

trait HasPinnedDropReceiver {
    fn drop(self: &pin mut Self);
}

impl HasPinnedDropReceiver for S {
    //~^ ERROR not all trait items implemented, missing: `drop` [E0046]
    fn drop(&pin mut self) {}
    //~^ ERROR method `drop` is not a member of trait `HasPinnedDropReceiver` [E0407]
}

struct Inherent;

impl Inherent {
    fn drop(&pin mut self) {}
}

mod local_drop_trait {
    use std::pin::Pin;

    struct S;

    trait Drop {
        fn pin_drop(self: Pin<&mut Self>);
    }

    impl Drop for S {
        //~^ ERROR not all trait items implemented, missing: `pin_drop` [E0046]
        fn drop(&pin mut self) {}
        //~^ ERROR method `drop` with `&pin mut self` is only supported for the `Drop` trait
    }
}

fn main() {}
