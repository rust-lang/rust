struct Wrapper<'rom>(T);
//~^ ERROR cannot find type `T` in this scope

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

fn main() {}
