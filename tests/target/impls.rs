impl Foo for Bar {
    fn foo() {
        "hi"
    }
}

pub impl Foo for Bar {
    // Comment 1
    fn foo() {
        "hi"
    }
    // Comment 2
    fn foo() {
        "hi"
    }
}

pub unsafe impl<'a, 'b, X, Y: Foo<Bar>> !Foo<'a, X> for Bar<'b, Y> where X: Foo<'a, Z>
{
    fn foo() {
        "hi"
    }
}

impl<'a, 'b, X, Y: Foo<Bar>> Foo<'a, X> for Bar<'b, Y>
    where X: Fooooooooooooooooooooooooooooo<'a, Z>
{
    fn foo() {
        "hi"
    }
}

impl<'a, 'b, X, Y: Foo<Bar>> Foo<'a, X> for Bar<'b, Y> where X: Foooooooooooooooooooooooooooo<'a, Z>
{
    fn foo() {
        "hi"
    }
}
