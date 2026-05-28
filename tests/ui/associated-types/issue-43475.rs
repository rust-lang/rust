//@ check-pass

trait Foo { type FooT: Foo; }
impl Foo for () { type FooT = (); }
trait Bar<T: Foo> { type BarT: Bar<T::FooT>; }
impl Bar<()> for () { type BarT = (); }

#[allow(dead_code)]
fn test<C: Bar<()>>() { }
fn main() { }
