trait Bar {
    type Type;
}
struct Foo<'a>(&'a ());
impl<'a> Bar for Foo<'f> { //~ ERROR undeclared lifetime
    type Type = u32;
}

fn test() //~ ERROR the trait bound `for<'a> Foo<'a>: Bar` is not satisfied
          //~| ERROR the trait bound `for<'a> Foo<'a>: Bar` is not satisfied
where
    for<'a> <Foo<'a> as Bar>::Type: Sized,
{
}

fn main() {}
