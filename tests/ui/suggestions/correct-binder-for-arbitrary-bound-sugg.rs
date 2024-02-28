trait Foo
where
    for<'a> &'a Self: Bar,
{
}

impl Foo for () {}

trait Bar {}

impl Bar for &() {}

fn foo<T: Foo>() {}
//~^ ERROR the trait `for<'a> Bar` is not implemented for `&'a T`

fn main() {}
