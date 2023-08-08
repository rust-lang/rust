// compile-flags: -Ztrait-solver=next
// check-pass

trait Foo<'a> {}
trait Bar<'a> {}

impl<'a, T: Bar<'a>> Foo<'a> for T {}
impl<T> Bar<'static> for T {}

fn main() {}
