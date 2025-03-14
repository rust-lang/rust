#![feature(precise_capturing_of_types)]
//~^ WARN the feature `precise_capturing_of_types` is incomplete

fn foo<T>(x: T) -> impl Sized + use<> {
    x
    //~^ ERROR hidden type mentions uncaptured type parameter `T`
}

fn main() {}
