//@ compile-flags: -Znext-solver
//@ check-pass

trait Foo<'a> {}
trait Bar<'a> {}

impl<'a, T: Bar<'a>> Foo<'a> for T {}
impl<T> Bar<'static> for T {}

fn main() {}
