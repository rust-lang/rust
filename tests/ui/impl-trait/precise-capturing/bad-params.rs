#![feature(precise_capturing)]
//~^ WARN the feature `precise_capturing` is incomplete

fn missing() -> impl use<T> Sized {}
//~^ ERROR cannot find type `T` in this scope

fn missing_self() -> impl use<Self> Sized {}
//~^ ERROR cannot find type `Self` in this scope

struct MyType;
impl MyType {
    fn self_is_not_param() -> impl use<Self> Sized {}
    //~^ ERROR `Self` can't be captured in `use<...>` precise captures list, since it is an alias
}

fn main() {}
