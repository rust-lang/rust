trait Foo {
    fn err(&self) -> MissingType;
    //~^ ERROR cannot find type `MissingType` in this scope
}

impl Foo for i32 {
    fn err(&self) -> MissingType {
        //~^ ERROR cannot find type `MissingType` in this scope
        0
    }
}

fn coerce(x: &i32) -> &dyn Foo {
    x
}

fn main() {}
