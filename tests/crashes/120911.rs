//@ known-bug: #120911
trait Container {
    type Item<'a>;
}
impl Container for () {
    type Item<'a> = ();
}
struct Exchange<C, F> {
    _marker: std::marker::PhantomData<(C, F)>,
}
fn exchange<C, F>(_: F) -> Exchange<C, F>
where
    C: Container,
    for<'a> F: FnMut(&C::Item<'a>),
{
    unimplemented!()
}
trait Parallelization<C> {}
impl<C, F> Parallelization<C> for Exchange<C, F> {}
fn unary_frontier<P: Parallelization<()>>(_: P) {}
fn main() {
    let exchange = exchange(|_| ());
    let _ = || {
        unary_frontier(exchange);
    };
}
