struct A<T>(B<T>);
//~^ ERROR recursive types `A` and `B` have infinite size
//~| ERROR `T` is never used
struct B<T>(A<A<T>>);
//~^ ERROR `T` is never used
trait Foo {}
impl<T> Foo for T where T: Send {}
impl Foo for B<u8> {}
//~^ ERROR conflicting implementations of trait `Foo` for type `B<u8>`
fn main() {}
