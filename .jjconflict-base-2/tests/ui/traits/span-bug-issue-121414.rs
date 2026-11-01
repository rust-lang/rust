trait Bar {
    type Type;
}
struct Foo<'a>(&'a ());
impl<'a> Bar for Foo<'f> { //~ ERROR undeclared lifetime
    type Type = u32;
}

fn test()
where
    for<'a> <Foo<'a> as Bar>::Type: Sized,
{
}

fn main() {}
