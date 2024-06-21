trait Foo {
    fn err(&self) -> MissingType;
    //~^ ERROR cannot find type `MissingType`
}

impl Foo for i32 {
    fn err(&self) -> MissingType {
        //~^ ERROR cannot find type `MissingType`
        0
    }
}

fn coerce(x: &i32) -> &dyn Foo {
    x
}

fn main() {}
