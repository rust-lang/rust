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

pub trait Shazam {}

pub fn bippety<X>() -> &'static X where X: Shazam {
    panic!()
}

pub struct Drizzel<T>(T);

impl<T> Drizzel<T> {
    pub fn boppety(&self) -> &T where T: Shazam {
        panic!();
    }
}
