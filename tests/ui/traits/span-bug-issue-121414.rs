trait Bar {
    type Type;
}
struct Foo<'a>(&'a ());
impl<'a> Bar for Foo<'f> { //~ ERROR undeclared lifetime
    type Type = u32;
}

fn test() //~ ERROR implementation of `Bar` is not general enough
where
    for<'a> <Foo<'a> as Bar>::Type: Sized,
{
}

fn main() {}
