// check-pass

trait Bar {
    type Type;
}
struct Foo<'a>(&'a ());
impl<'a> Bar for Foo<'a> {
    type Type = ();
}

fn func<'a>(_: <Foo<'a> as Bar>::Type) {}
fn assert_is_func<A>(_: fn(A)) {}

fn test()
where
    for<'a> <Foo<'a> as Bar>::Type: Sized,
{
    assert_is_func(func);
}

fn main() {}
