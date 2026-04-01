//@ known-bug: #136666
// Needed so that rust can infer that the A in what() is &()
trait IsRef<T> {}
struct Dummy;
impl<'a> IsRef<&'a ()> for Dummy {}

trait WithLifetime {
    type Output<'a>;
}
impl<'t> WithLifetime for &'t () {
    type Output<'a> = &'a ();
}

// Needed to prevent the two Foo impls from overlapping
struct Wrap<A>(A);

trait Unimplemented {}

trait Foo {}
impl<T> Foo for T where T: Unimplemented {}
impl<A> Foo for Wrap<A>
where
    Dummy: IsRef<A>,
    for<'a> A: WithLifetime<Output<'a> = A>,
{
}

fn what<A>()
where
    Wrap<A>: Foo,
{
}

fn main() {
    what();
}
