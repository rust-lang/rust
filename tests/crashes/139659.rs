//@ known-bug: #139659
//@compile-flags: -Cdebuginfo=2 -Copt-level=0 --crate-type lib
trait Trait {
    type Output;
}

impl<O, F: Fn() -> O> Trait for F {
    type Output = O;
}

struct Wrap<P>(P);
struct WrapOutput<O>(O);

impl<P: Trait> Trait for Wrap<P> {
    type Output = WrapOutput<P::Output>;
}

fn wrap<P: Trait>(x: P) -> impl Trait {
    Wrap(x)
}

fn consume<P: Trait>(_: P) -> P::Output {
    unimplemented!()
}

pub fn recurse() -> impl Sized {
    consume(wrap(recurse))
}
pub fn main() {}
