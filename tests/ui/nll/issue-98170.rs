pub struct MyStruct<'a> {
    field: &'a [u32],
}

impl MyStruct<'_> {
    pub fn new<'a>(field: &'a [u32]) -> MyStruct<'a> {
        Self { field }
        //~^ ERROR lifetime may not live long enough
        //~| ERROR lifetime may not live long enough
    }
}

trait Trait<'a> {
    fn new(field: &'a [u32]) -> MyStruct<'a>;
}

impl<'a> Trait<'a> for MyStruct<'_> {
    fn new(field: &'a [u32]) -> MyStruct<'a> {
        Self { field }
        //~^ ERROR lifetime may not live long enough
        //~| ERROR lifetime may not live long enough
    }
}

fn main() {}
