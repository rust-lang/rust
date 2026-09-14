//@ known-bug: #147719
//@ edition: 2024
trait NodeImpl {}
struct Wrap<F, P>(F, P);
impl<F, P> Wrap<F, P> {
    fn new(_: F) -> Self {
        loop {}
    }
}
trait Arg {}
impl<F, A> NodeImpl for Wrap<F, A> where A: Arg {}
impl<F, Fut, A> NodeImpl for Wrap<F, (A,)> where F: Fn(&(), A) -> Fut {}
fn trigger_ice() {
    let _: &dyn NodeImpl = &Wrap::<_, (i128,)>::new(async |_: &(), i128| 0);
}
fn main() {}
