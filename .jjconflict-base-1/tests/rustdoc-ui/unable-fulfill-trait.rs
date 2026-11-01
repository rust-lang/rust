// This test ensures that it's not crashing rustdoc.

pub struct Foo<'a, 'b, T> {
    field1: dyn Bar<'a, 'b>,
    //~^ ERROR
}

pub trait Bar<'x, 's, U>
where
    U: 'x,
    Self: 'x,
    Self: 's,
{
}
