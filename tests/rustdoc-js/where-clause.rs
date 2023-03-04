pub struct Nested;

pub trait Trait<T> {
    fn thank_you(x: T);
}

pub fn abracadabra<X>(_: X) where X: Trait<Nested> {}

pub fn alacazam<X>() -> X where X: Trait<Nested> {}

pub trait T1 {}
pub trait T2<'a, T> {
    fn please(_: &'a T);
}

pub fn presto<A, B>(_: A, _: B) where A: T1, B: for <'b> T2<'b, Nested> {}
