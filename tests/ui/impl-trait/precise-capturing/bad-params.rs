fn missing() -> impl Sized + use<T> {}
//~^ ERROR cannot find type or const parameter `T` in this scope

fn missing_self() -> impl Sized + use<Self> {}
//~^ ERROR cannot find type or const parameter `Self` in this scope

struct MyType;
impl MyType {
    fn self_is_not_param() -> impl Sized + use<Self> {}
    //~^ ERROR `Self` can't be captured in `use<...>` precise captures list, since it is an alias
}

fn hello() -> impl Sized + use<hello> {}
//~^ ERROR expected type or const parameter, found function `hello`

fn arg(x: ()) -> impl Sized + use<x> {}
//~^ ERROR expected type or const parameter, found local variable `x`

fn main() {}
