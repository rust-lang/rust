//~ ERROR overflow evaluating the requirement `A<A<A<A<A<A<A<...>>>>>>>: Send`
struct A<T>(B<T>);
//~^ ERROR recursive types `A` and `B` have infinite size
struct B<T>(A<A<T>>);
trait Foo {}
impl<T> Foo for T where T: Send {}
impl Foo for B<u8> {}

fn main() {}
