trait Foo<'a> {}

fn or<'a>(first: &'static dyn Foo<'a>) -> dyn Foo<'a> {
    //~^ ERROR return type cannot be a trait object without pointer indirection
    return Box::new(panic!());
}

fn main() {}
