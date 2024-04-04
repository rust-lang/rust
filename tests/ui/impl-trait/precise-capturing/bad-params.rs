#![feature(precise_capturing)]
//~^ WARN the feature `precise_capturing` is incomplete

fn missing() -> impl use<T> Sized {}
//~^ ERROR could not find type or const parameter

fn missing_self() -> impl use<Self> Sized {}
//~^ ERROR could not find type or const parameter

struct MyType;
impl MyType {
    fn self_is_not_param() -> impl use<Self> Sized {}
    //~^ ERROR `Self` cannot be captured because it is not a type parameter
}

fn main() {}
