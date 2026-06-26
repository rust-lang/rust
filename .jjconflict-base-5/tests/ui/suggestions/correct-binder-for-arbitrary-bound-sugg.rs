trait Foo
where
    for<'a> &'a Self: Bar,
{
}

impl Foo for () {}

trait Bar {}

impl Bar for &() {}

fn foo<T: Foo>() {}
//~^ ERROR the trait bound `for<'a> &'a T: Bar` is not satisfied

fn main() {}
