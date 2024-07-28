//~ ERROR overflow evaluating the requirement `A<A<A<A<A<A<A<...>>>>>>>: Send`
struct A<T>(B<T>);
//~^ ERROR recursive types `A` and `B` have infinite size
//~| ERROR `T` is only used recursively
struct B<T>(A<A<T>>);
//~^ ERROR `T` is only used recursively
trait Foo {}
impl<T> Foo for T where T: Send {}
impl Foo for B<u8> {}

fn main() {}
