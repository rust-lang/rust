trait NoLifetime {
    fn get<'p, T : Test<'p>>(&self) -> T;
    //~^ NOTE lifetimes in impl do not match this method in trait
}

trait Test<'p> {
    fn new(buf: &'p mut [u8]) -> Self;
}

struct Foo<'a> {
    buf: &'a mut [u8],
}

impl<'a> Test<'a> for Foo<'a> {
    fn new(buf: &'a mut [u8]) -> Foo<'a> {
        Foo { buf: buf }
    }
}

impl<'a> NoLifetime for Foo<'a> {
    fn get<'p, T : Test<'a>>(&self) -> T {
    //~^ ERROR E0195
    //~| NOTE lifetimes do not match method in trait
        return *self as T;
        //~^ ERROR non-primitive cast: `Foo<'a>` as `T`
        //~| NOTE an `as` expression can only be used to convert between primitive types.
    }
}

fn main() {}
