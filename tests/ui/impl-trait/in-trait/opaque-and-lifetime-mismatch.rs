struct Wrapper<'rom>(&'rom ());

trait Foo {
    fn bar() -> Wrapper<impl Sized>;
    //~^ ERROR missing lifetime specifier
    //~| ERROR struct takes 0 generic arguments but 1 generic argument was supplied
}

impl Foo for () {
    fn bar() -> i32 {
        //~^ ERROR method `bar` has an incompatible type for trait
        //~| ERROR method `bar` has an incompatible return type for trait
        0
    }
}

trait Bar {
    fn foo() -> Wrapper<impl Sized>;
    //~^ ERROR missing lifetime specifier
    //~| ERROR struct takes 0 generic arguments but 1 generic argument was supplied
}

impl Bar for () {
    fn foo() -> Wrapper<impl Sized> {
        //~^ ERROR missing lifetime specifier
        //~| ERROR struct takes 0 generic arguments but 1 generic argument was supplied
        Wrapper(&())
    }
}

fn main() {}
