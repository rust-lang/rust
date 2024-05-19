trait Foo<'a> {}

fn or<'a>(first: &'static dyn Foo<'a>) -> dyn Foo<'a> {
    //~^ ERROR return type cannot have an unboxed trait object
    return Box::new(panic!());
}

fn main() {}
