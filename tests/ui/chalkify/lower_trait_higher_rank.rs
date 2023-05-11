// check-pass
// compile-flags: -Z trait-solver=chalk

trait Foo<F: ?Sized> where for<'a> F: Fn(&'a (u8, u16)) -> &'a u8
{
}

fn main() {
}
